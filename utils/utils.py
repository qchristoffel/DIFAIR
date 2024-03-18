import datetime
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import logging
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve


def get_time():
    return time.time()


def exec_time(start_time):
    return datetime.timedelta(seconds=time.time() - start_time)


def plot_history(history, range=None, return_fig=False):
    if type(history) == tf.keras.callbacks.History:
        history = history.history

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))

    if not range:
        sl = np.s_[:]
    else:
        sl = np.s_[range[0] : range[1]]

    # plot loss
    ax0.plot(history["loss"][sl], label="loss")
    ax0.plot(history["val_loss"][sl], label="val loss")
    ax0.plot(history["test_loss"][sl], label="test loss")
    ax0.set_xlabel("Epochs")
    ax0.set_ylabel("Loss")
    ax0.legend()

    # plot accuracy
    for key in history.keys():
        validation = False
        if "accuracy" in key:
            if "val" in key:
                validation = True
            if type(history[key]) == dict:
                for k, v in history[key].items():
                    ax1.plot(v[sl], label="val" + k if validation else k)
            else:
                ax1.plot(history[key][sl], label="val " + key if validation else key)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    if return_fig:
        return fig
    else:
        plt.show()


def plot_auroc_from_history(history, range=None, return_fig=False):
    if type(history) == tf.keras.callbacks.History:
        history = history.history

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    if not range:
        sl = np.s_[:]
    else:
        sl = np.s_[range[0] : range[1]]

    ax.plot(history["val_max_val_auroc"][sl], label="max val auroc")
    ax.plot(history["val_real_auroc"][sl], label="real auroc")
    ax.legend()

    if return_fig:
        return fig
    else:
        plt.show()


def create_dir(path):
    if not tf.io.gfile.exists(path):
        logging.info(f"Creating directory {path}")
        tf.io.gfile.makedirs(path)


def check_save_path(FLAGS, recovering=False):
    path = os.path.join(FLAGS.save_path, FLAGS.prefix)
    already_exists = []
    for extension in ["model.save", "history.pkl", "flags.txt"]:
        if tf.io.gfile.exists(path + extension):
            already_exists.append(True)
        else:
            already_exists.append(False)

    if recovering:
        if not all(already_exists):
            logging.info(f"Recovering run at {path}, it might have been interrupted.")
        elif all(already_exists):
            logging.info(f"This run has already been completed and saved at {path}.")
            exit(0)
    else:
        if any(already_exists):
            raise ValueError(
                f"""Some save files at {path} already exists. Your run would overwrite them.
                Change prefix or delete the file."""
            )


def save_flags(FLAGS):
    if FLAGS.save_path:
        path = os.path.join(FLAGS.save_path, FLAGS.prefix)

        logging.info(f"Saving flags in {FLAGS.save_path}/{FLAGS.prefix}flags.txt")
        # FLAGS.append_flags_into_file(path + "flags.txt")

        key_flags_user = FLAGS.get_key_flags_for_module(sys.argv[0])
        s = "\n".join(f.serialize() for f in key_flags_user) + "\n"
        key_flags_module = FLAGS.get_key_flags_for_module("config_flags")
        s += "\n".join(f.serialize() for f in key_flags_module)

        with open(path + "flags.txt", "w") as f:
            f.write(s)


def save_results(base, history_base, FLAGS):
    # need the defined flags object
    if FLAGS.save_path and FLAGS.to_save != ["none"]:
        path = os.path.join(FLAGS.save_path, FLAGS.prefix)

        if "all" in FLAGS.to_save or "model" in FLAGS.to_save:
            logging.info(f"Saving model in {FLAGS.save_path}/{FLAGS.prefix}model.save")
            base.save(path + "model.save")

        if "all" in FLAGS.to_save or "history" in FLAGS.to_save:
            logging.info(
                f"Saving history in {FLAGS.save_path}/{FLAGS.prefix}history.pkl"
            )
            pickle.dump(history_base, open(path + "history.pkl", "wb"))


def plot_roc_curve(pred_known, pred_unknown, loss_helper, save=None, plot=False):
    score_known = loss_helper.osr_score(pred_known)
    score_unknown = loss_helper.osr_score(pred_unknown)
    y_true, y_pred = loss_helper._format_score(score_known, score_unknown)

    fpr, tpr, _ = roc_curve(y_true, y_pred)

    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot()

    if save:
        print("Saving roc curve.")
        plt.savefig(save)

    if plot:
        plt.show()


def actualize_centers(
    original_model,
    feature_extractor,
    images,
    labels,
    loss_helper,
    nb_classes,
    batch_size=256,
):
    """Using 'original model', indices of correctly classified samples will be
    found (need 'loss_helper' for that). Features are then extracted with the
    'feature_extractor' and only features of correctly classified samples are
    kept. New centers are the mean of these features for each class.
    """

    # predict with real model to know which are the correct predictions
    preds = original_model.predict(images, batch_size=batch_size)
    predicted_label = loss_helper.predicted_class(preds)
    correct_preds_indices = predicted_label == labels
    correct_labels = labels[correct_preds_indices]

    # compute new centers from features of correctly predicted samples
    features = feature_extractor.predict(images, batch_size=batch_size)
    correct_features = features[correct_preds_indices]

    class_centers = []
    for i in range(nb_classes):
        class_centers.append(
            tf.reduce_mean(correct_features[correct_labels == i], axis=0)
        )

    return class_centers


@tf.function
def tf_correlation(x):
    # self correlation of x

    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = x - mean_x

    std_x = tf.math.reduce_std(x, axis=0, keepdims=True)
    std_x = tf.where(std_x == 0, tf.ones_like(std_x), std_x)

    cov_x = tf.matmul(mx, mx, transpose_a=True) / x.shape[0]

    # tf.summary.scalar("cov_x", tf.reduce_mean(cov_x))
    # tf.summary.scalar("std_x", tf.reduce_mean(std_x))

    return cov_x / tf.matmul(std_x, std_x, transpose_a=True)


class TensorboardWriters:
    def __init__(self, writer_types, args):
        self.writer_types = writer_types
        self.current_writer = None

        self.init_writers(args.tensorboard, args.prefix)

    def init_writers(self, directory, prefix):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.writers = {}
        for writer in self.writer_types:
            log_dir = directory + "/" + prefix + current_time + "/" + writer
            self.writers[writer] = tf.summary.create_file_writer(
                log_dir, experimental_trackable=True
            )

    def get_writer(self, writer):
        if writer not in self.writer_types:
            raise ValueError(f"Writer type {writer} not in {self.writer_types}")
        return self.writers[writer]
