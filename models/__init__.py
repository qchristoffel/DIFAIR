import numpy as np
import tensorflow as tf

from metrics.metrics import weights_convergence_metric, weights_correlation_metric
from metrics.osr import evaluation


class Model:
    def __init__(
        self,
        model,
        optimizer,
        loss_helper,
        metrics,
        weights_penalties,
        parallel_strategy,
        global_batch_size,
        nb_batches,
        verbose=1,
        writers=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_helper = loss_helper
        self.parallel_strategy = parallel_strategy
        self.nb_batches = nb_batches
        self.verbose = verbose
        self.writers = writers  # must be a dict with keys 'train', 'val' and 'test'

        with self.parallel_strategy.scope():

            def compute_loss(labels, predictions, model_losses):
                per_example_loss = self.loss_helper.loss(
                    labels, predictions, write=self.writers is not None
                )
                loss = tf.nn.compute_average_loss(
                    per_example_loss, global_batch_size=global_batch_size
                )
                if model_losses:
                    ml = tf.nn.scale_regularization_loss(tf.add_n(model_losses))
                    loss += ml  # TODO : test difference when not added to loss
                else:
                    ml = 0.0
                return loss, ml

            self.compute_loss = compute_loss

            self.test_loss = tf.keras.metrics.Mean(name="val_loss")
            self.model_losses = tf.keras.metrics.Mean(name="model_losses")

        self.metrics = metrics
        self.weights_penalties_fn = weights_penalties
        self.__init_history()

    # --------------------------------------------------------------------------#
    # Metrics

    def update_metrics(self, y_true, y_pred):
        for metric in self.metrics:
            if self.loss_helper.reconstruction:
                pred_label = self.loss_helper.predicted_class(y_pred[0])
            else:
                pred_label = self.loss_helper.predicted_class(y_pred)
            metric.update_state(y_true, pred_label)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def get_metrics(self):
        return {m.name: m.result().numpy() for m in self.metrics}
        # if type(m) != dict
        # else {k: v.dtype() for k, v in m.items()}
        # for m in self.metrics}

    def __init_history(self):
        res = self.get_metrics()

        self.history = {}
        for key, val in res.items():
            if type(val) == dict:
                self.history[key] = {k: [] for k in val}
                self.history["val_" + key] = {k: [] for k in val}
                self.history["test_" + key] = {k: [] for k in val}
            else:
                self.history[key] = []
                self.history["val_" + key] = []
                self.history["test_" + key] = []

        self.history["loss"] = []
        self.history["model_losses"] = []
        self.history["mean_weights_std"] = []
        self.history["mean_weights_corr"] = []
        self.history["weights_sample"] = []

        # For validation
        self.history["val_loss"] = []
        self.history["val_real_auroc"] = []
        self.history["val_max_val_auroc"] = []
        self.history["val_oscr"] = []

        # For test
        self.history["test_loss"] = []
        self.history["test_real_auroc"] = []
        self.history["test_max_val_auroc"] = []
        self.history["test_oscr"] = []

    def _update_history(
        self,
        train_loss,
        train_metrics,
        val_loss,
        val_metrics,
        osr_results,
    ):
        # Losses
        self.history["loss"].append(train_loss.numpy())
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)

        self.history["model_losses"].append(self.model_losses.result().numpy())

        # Osr results
        if osr_results != None:
            self.history["val_real_auroc"].append(osr_results["real_auroc"])
            self.history["val_max_val_auroc"].append(osr_results["max_val_auroc"])
            self.history["val_oscr"].append(osr_results["oscr"])

        # Other metrics
        def _update_dict(metrics, prefix):
            for key, val in metrics.items():
                if type(val) == dict:
                    for k, v in val.items():
                        self.history[prefix + key][k].append(v)
                else:
                    self.history[prefix + key].append(val)

        _update_dict(train_metrics, prefix="")
        if val_metrics is not None:
            _update_dict(val_metrics, prefix="val_")

    def _update_history_with_test(
        self,
        test_loss,
        test_metrics,
        test_osr_results,
    ):
        def _update_dict(metrics, prefix):
            for key, val in metrics.items():
                if type(val) == dict:
                    for k, v in val.items():
                        self.history[prefix + key][k].append(v)
                else:
                    self.history[prefix + key].append(val)

        # keep appending for backward compatibility
        # (TODO: change this everywhere)
        self.history["test_loss"].append(test_loss)
        _update_dict(test_metrics, prefix="test_")

        if test_osr_results != None:
            self.history["test_real_auroc"].append(test_osr_results["real_auroc"])
            self.history["test_max_val_auroc"].append(test_osr_results["max_val_auroc"])
            self.history["test_oscr"].append(test_osr_results["oscr"])

    # --------------------------------------------------------------------------#
    # Training
    def train_step(self, inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss, ml = self.compute_loss(labels, predictions, self.model.losses)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.model_losses.update_state(ml)
        self.update_metrics(labels, predictions)
        return loss

    @tf.function
    def distributed_train_step(self, dataset_inputs):
        per_replica_losses = self.parallel_strategy.run(
            self.train_step, args=(dataset_inputs,)
        )
        # reduce metrics if dict ?
        return self.parallel_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    # --------------------------------------------------------------------------#
    # Evaluation
    def test_step(self, inputs):
        images, labels = inputs

        predictions = self.model(images, training=False)
        t_loss = self.loss_helper.loss(labels, predictions)

        self.test_loss.update_state(t_loss)
        self.update_metrics(labels, predictions)

    @tf.function
    def distributed_test_step(self, dataset_inputs):
        return self.parallel_strategy.run(self.test_step, args=(dataset_inputs,))

    def test(self, ds_test_dict):
        self.reset_metrics()
        self.test_loss.reset_states()

        for x in ds_test_dict:
            self.distributed_test_step(x)

        metrics_dict = self.get_metrics()
        metrics = {
            metric: val if type(val) != dict else {k: v for k, v in val.items()}
            for metric, val in metrics_dict.items()
        }

        return self.test_loss.result().numpy(), metrics

    # --------------------------------------------------------------------------#
    # Prediction
    def predict_step(self, inputs, compute_metrics=False):
        images, labels = inputs
        predictions = self.model(images, training=False)

        if compute_metrics:
            t_loss = self.loss_helper.loss(labels, predictions)

            self.test_loss.update_state(t_loss)
            self.update_metrics(labels, predictions)

        return predictions

    @tf.function
    def distributed_prediction_step(self, inputs, compute_metrics=False):
        # Perform the forward pass
        predictions = self.parallel_strategy.run(
            self.predict_step, args=(inputs, compute_metrics)
        )

        # Return predictions
        return self.parallel_strategy.experimental_local_results(predictions)

    def predict(self, ds, compute_metrics=False):
        # Returns a list of predictions for each output of the model. Therefore
        # if there is only one output, then the output is a list with one element

        # If compute metrics is not false, then compute metrics and store them
        # self.test_loss
        if compute_metrics:
            self.reset_metrics()
            self.test_loss.reset_states()

        if self.loss_helper.reconstruction:
            nb_outputs = 3  # preds, prev layer output, prev layer reconstruction
        else:
            nb_outputs = 1
        all_predictions = []
        for _ in range(nb_outputs):
            all_predictions.append([])

        for x in ds:
            predictions = self.distributed_prediction_step(x, compute_metrics)

            if nb_outputs == 1:
                all_predictions[0].extend(predictions)
            else:
                for gpu in predictions:
                    for i, pred in enumerate(gpu):
                        all_predictions[i].append(pred)

        result = []
        for preds in all_predictions:
            result.append(tf.concat(preds, axis=0).numpy())

        return result

    # --------------------------------------------------------------------------#
    # Training loop
    def train(self, datasets, epochs):
        ds_train_dist = datasets["ds_train_known"]

        # Train
        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.reset_metrics()
            self.test_loss.reset_states()
            self.model_losses.reset_states()
            if self.writers is not None:
                self.writers.get_writer("train").set_as_default(
                    self.optimizer.iterations
                )

            total_loss = 0.0
            n_batch = 0

            tf.print(f"\nEpoch {epoch + 1:>2}/{epochs:<3}")
            tf.print("Learning rate:", self.optimizer.lr)

            progbar = tf.keras.utils.Progbar(
                self.nb_batches,
                stateful_metrics=["loss", "val_loss"],
                verbose=self.verbose,
            )

            for x in ds_train_dist:
                n_batch += 1
                total_loss += self.distributed_train_step(x)

                progbar.update(
                    n_batch,
                    values=[
                        ("loss", (total_loss / tf.cast(n_batch, dtype=tf.float32)))
                    ],
                    finalize=False,
                )

            train_loss = total_loss / tf.cast(n_batch, dtype=tf.float32)
            tf.summary.scalar("loss", train_loss, step=epoch)

            train_metrics = self.get_metrics()
            prog_values = []
            for k in train_metrics:
                prog_values.append((k, train_metrics[k]))
                tf.summary.scalar(k, train_metrics[k], step=epoch)

            # weights penalties (std and corr)
            for key, fn in self.weights_penalties_fn.items():
                tf.summary.scalar("loss/" + key, fn(), step=epoch)

            prog_values.insert(0, ("loss", train_loss))
            progbar.update(n_batch, values=prog_values, finalize=True)

            # Compute metrics on weights
            self.save_weights_metrics(epoch)

            # Save some weights value to check evolution
            self.save_some_weights()

            # Validation
            if datasets["ds_val_known"] is not None:
                val_loss, val_metrics, results = self.evaluate_osr(
                    datasets["ds_val_known"],
                    datasets["ds_val_unknown"],
                    "val",
                    epoch,
                )
            else:
                val_loss, val_metrics, results = None, None, None

            self._update_history(
                train_loss,
                train_metrics,
                val_loss,
                val_metrics,
                results,
            )
            print()

        # Test
        test_loss, test_metrics, results = self.evaluate_osr(
            datasets["ds_test_known"],
            datasets["ds_test_unknown"],
            "test",
            epoch,
        )

        self._update_history_with_test(test_loss, test_metrics, results)

        return self.history

    def save_some_weights(self):
        try:
            layer = self.model.get_layer("last_conv")
        except:
            layer = None

        if layer is not None:
            weights = layer.get_weights()[0]
            selected_weights = weights[0, 0, 0, :]
            self.history["weights_sample"].append(selected_weights)

    def save_weights_metrics(self, epoch):
        try:
            layer = self.model.get_layer("last_conv")
            weights = layer.get_weights()[0]
        except:
            weights = None

        # Compute weights convergence metric
        mean_weights_std = None
        if self.loss_helper.distance_based and weights is not None:

            mean_weights_std = weights_convergence_metric(
                weights,
                self.loss_helper.nb_classes,
                self.loss_helper.nb_features,
            )
            tf.summary.scalar(
                "mean_weights_std", tf.reduce_mean(mean_weights_std), step=epoch
            )
            print("Weights convergence metric (std):", mean_weights_std.numpy())

            self.history["mean_weights_std"].append(mean_weights_std.numpy())

        # Compute weights correlation metric
        if weights is not None:
            mean_weights_corr = weights_correlation_metric(
                weights,
                self.loss_helper.nb_classes,
                self.loss_helper.nb_features,
            )
            tf.summary.scalar("mean_weights_corr", mean_weights_corr, step=epoch)

            self.history["mean_weights_corr"].append(mean_weights_corr)

    def evaluate_osr(self, dataset_known, dataset_unknown, mode, epoch):
        self.reset_metrics()
        if self.writers is not None and mode == "val":
            self.writers.get_writer(mode).set_as_default(epoch)

        if mode == "val":
            print("Validation results:")
        elif mode == "test":
            print("Test results:")

        preds_known = self.predict(dataset_known, compute_metrics=True)[0]

        # Loss and metrics were computed by predict with compute_metrics=True
        loss = self.test_loss.result().numpy()
        if mode == "val":
            tf.summary.scalar("loss", loss)
        print(f"\tLoss: {loss}\n\t", end="")

        metrics = self.get_metrics()
        for k, v in metrics.items():
            print(f"{k}: {v}", end=" ")
            tf.summary.scalar(k, v)
        print()

        if dataset_unknown is not None:
            preds_unknown = self.predict(dataset_unknown)[0]

            labels = np.array([labels for _, labels in dataset_known.unbatch()])
            results = evaluation(preds_known, preds_unknown, labels, self.loss_helper)
            for k, v in results.items():
                tf.summary.scalar("osr/" + k, v)
        else:
            results = None

        return loss, metrics, results
