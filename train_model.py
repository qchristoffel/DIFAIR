import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.allow_growth = True

from pprint import pprint

from absl import app, flags

import config_flags
import models.architectures as arch
import utils.utils as u
from datasets.data import load_dataset
from helpers import get_loss_helper
from losses.losses import add_last_conv_penalties
from metrics.evaluation import hypersphere_percentage
from models import Model
from utils.schedulers import get_scheduler

# Special option for running on cluster where jobs can be cancelled.
# This forces the program to be rerun if all save files are not present.
RECOVERING = True

NGPUS = len(tf.config.list_physical_devices("GPU"))
# define strategy for multi-gpu training
if NGPUS <= 1:
    parallel_strategy = tf.distribute.get_strategy()
elif NGPUS > 1:
    parallel_strategy = tf.distribute.MirroredStrategy()

FLAGS = flags.FLAGS


class SGD_reworked(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="SGD",
        **kwargs,
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs,
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

    def build(self, var_list):
        """Initialize optimizer variables.

        SGD optimizer has one variable `momentums`, only set if `self.momentum`
        is not 0.

        Args:
          var_list: list of model variables to build SGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(model_variable=var, variable_name="m")
            )
        self._built = True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        m = None
        var_key = self._var_key(variable)
        momentum = tf.cast(self.momentum, variable.dtype)
        m = self.momentums[self._index_dict[var_key]]

        # Dense gradients
        if m is not None:
            m.assign(m * momentum + gradient)
            if self.nesterov:
                variable.assign_add(-gradient * lr + m * momentum)
            else:
                variable.assign_add(-lr * m)
        else:
            variable.assign_add(-gradient * lr)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
            }
        )
        return config


def main(argv):
    # Set seed for reproducibility
    tf.keras.utils.set_random_seed(FLAGS.seed)

    # Need to take care of the case where the last element is "none" because
    # 'to_save' is a multi enum, so adding flags do not replace previous ones.
    # Ff the last element is "none" then save nothing
    if FLAGS.to_save[-1] == "none":
        FLAGS.to_save = [FLAGS.to_save[-1]]

    if FLAGS.to_save != ["none"] and FLAGS.save_path:
        u.check_save_path(FLAGS, RECOVERING)
        u.create_dir(FLAGS.save_path)
        u.save_flags(FLAGS)

    global_batch_size = FLAGS.batch_size * parallel_strategy.num_replicas_in_sync

    start_time = u.get_time()
    datasets, nb_classes, nb_batches, nb_channels, norm_layer, _ = load_dataset(
        FLAGS, parallel_strategy=parallel_strategy
    )
    print("--- Data preprocessing time : %s ---" % (u.exec_time(start_time)))

    class_anchors = tf.repeat(tf.eye(nb_classes), FLAGS.nb_features, axis=1)
    class_anchors *= FLAGS.anchor_multiplier

    if FLAGS.noise_data > 0:
        # null anchor for noisy samples
        class_anchors = tf.concat(
            [class_anchors, tf.zeros((1, nb_classes * FLAGS.nb_features))], axis=0
        )

    with parallel_strategy.scope():
        # Get model
        model = arch.get_model(FLAGS, nb_classes, nb_channels, norm_layer)

        if FLAGS.nb_features > 1:
            # if there is only one feature no need to penalize the weights

            # Add penalties to last conv layer (avoid weights convergence)
            weights_penalties_fn = add_last_conv_penalties(
                model.get_layer("last_conv"), nb_classes, FLAGS
            )
        else:
            weights_penalties_fn = {}

        if FLAGS.summary:
            model.summary()
            # tf.keras.utils.plot_model(model, show_shapes=True)
            # exit()

        scheduler = get_scheduler(FLAGS, nb_batches)

        # optimizer = tf.keras.optimizers.Adam(scheduler)

        optimizer = SGD_reworked(
            learning_rate=scheduler, momentum=0.9, weight_decay=1e-4, jit_compile=False
        )

        # optimizer = keras.optimizers.experimental.SGD(
        #     learning_rate=scheduler,
        #     momentum=0,
        #     weight_decay=1e-4
        # )

        # optimizer = get_optimizer(FLAGS, scheduler)

        loss_helper = get_loss_helper(FLAGS, class_anchors, nb_classes)

        metrics = [tf.keras.metrics.Accuracy(name="accuracy")]

        if FLAGS.tensorboard is not None:
            writer_types = ["train", "test"]
            if FLAGS.split_train_val:
                writer_types.append("val")
            writers = u.TensorboardWriters(writer_types, FLAGS)
        else:
            writers = None

        # model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
        # history = model.fit(datasets["ds_train_known"], epochs=FLAGS.epochs, verbose=FLAGS.verbose)

        model_object = Model(
            model,
            optimizer=optimizer,
            loss_helper=loss_helper,
            metrics=metrics,
            weights_penalties=weights_penalties_fn,
            parallel_strategy=parallel_strategy,
            global_batch_size=global_batch_size,
            nb_batches=nb_batches,
            verbose=FLAGS.verbose,
            writers=writers,
        )

        start_time = u.get_time()
        history = model_object.train(datasets, FLAGS.epochs)
    print("--- Execution time : %s ---" % (u.exec_time(start_time)))

    if datasets["ds_val_known"] is not None and FLAGS.loss != "crossentropy":
        labels = np.array([labels for _, labels in datasets["ds_val_known"].unbatch()])

        if FLAGS.reconstruction:
            sl = np.s_[0]
        else:
            sl = np.s_[:]

        hp = hypersphere_percentage(
            model.predict(datasets["ds_val_known"])[sl],
            labels,
            class_anchors,
            FLAGS.max_dist,
        )
        history["val_hypersphere_percentage"] = [hp]
        print("Hypersphere percentage:", hp)

    # Evaluate learning on test set
    # test_scores = model.evaluate(datasets["ds_test_known"])
    test_scores = model_object.test(datasets["ds_test_known"])
    print("Closed set loss:", test_scores[0])
    print("Closed set metrics:")
    for k, v in test_scores[1].items():
        print(f"{k}: {v}", end=" ")
    print()

    ############################################################################
    def plot_weights_correlations():
        if FLAGS.loss == "difair":
            layer = model.get_layer("last_conv").get_weights()[0]
        else:
            layer = None
            for l in model.layers[::-1]:
                if "conv2d" in l.name:
                    layer = l.get_weights()[0]
                    break

        coeffs = u.tf_correlation(layer.reshape(-1, layer.shape[-1])).numpy()
        # coeffs = np.corrcoef(layer.reshape(-1, layer.shape[-1]), rowvar=False)
        # print(coeffs[:5, :5])

        # print(coeffs[5:10, 5:10])

        print(
            "mean correlation:",
            np.mean(np.abs(coeffs[np.triu_indices_from(coeffs, k=1)])),
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(coeffs, cmap="coolwarm")
        plt.title("Correlation matrix of weights of last convolutional layer")
        plt.colorbar(shrink=0.6)
        # plt.show()

        if "all" in FLAGS.to_save and FLAGS.save_path:
            plt.savefig(f"{FLAGS.save_path}/{FLAGS.prefix}weights_correlation.png")

    plot_weights_correlations()

    ############################################################################

    if datasets["ds_test_unknown"] is not None:
        # OSR evaluation
        y_pred_known = model.predict(datasets["ds_test_known"])
        y_pred_unknown = model.predict(datasets["ds_test_unknown"])

        if FLAGS.reconstruction:
            y_pred_known = y_pred_known[0]
            y_pred_unknown = y_pred_unknown[0]

        print("AUROC score:", loss_helper.auroc(y_pred_known, y_pred_unknown))

        save_roc = (
            f"{FLAGS.save_path}/{FLAGS.prefix}roc_curve.png"
            if "all" in FLAGS.to_save and FLAGS.save_path
            else None
        )
        u.plot_roc_curve(y_pred_known, y_pred_unknown, loss_helper, save=save_roc)

    # Save model
    u.save_results(model, history, FLAGS)

    # Model losses
    plt.figure()
    plt.plot(model_object.history["model_losses"])
    plt.title("Loss on weights : higher means weights are more similar")
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    if "all" in FLAGS.to_save and FLAGS.save_path:
        plt.savefig(f"{FLAGS.save_path}/{FLAGS.prefix}model_losses.png")

    # pprint(history)
    if FLAGS.plot:
        u.plot_history(history)
        plt.show()  # show all figures


if __name__ == "__main__":
    app.run(main)
