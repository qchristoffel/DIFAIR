# example usage:

# python3 analyze_results_model.py --flagfile results/tests_weight_loss/coef_2_3/coef_2_3_flags.txt --analyse output --plot_anchors --actualize_centers --save_format pdf
# python3 analyze_results_model.py --flagfile results/benchmark_osr/standard_vgg32/dist/cifar10/split-2/nb_f-5/anchor_mul-10/max_d-12.649/flags.txt --analyse output --plot_anchors --actualize_centers --save_format pdf
# python3 analyze_results_model.py --flagfile results/crossentropy_features/30features_flags.txt --analyse representation --plot_anchors --actualize_centers --save_format pdf

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import config_flags
import helpers
import utils.utils as u
import visualization as viz
from datasets.data import load_dataset

tf.get_logger().setLevel("ERROR")

flags.DEFINE_string(
    "analyse",
    default="representation",
    help="Whether to plot the representation or the output.",
)
flags.DEFINE_boolean(
    "plot_anchors",
    default=False,
    help="Whether to plot the anchors on tsne plots or not.",
)
flags.DEFINE_boolean(
    "actualize_centers", default=False, help="Whether to compute new centers or not."
)
flags.DEFINE_string("save_format", default="png", help="Format to save figures in.")
flags.DEFINE_boolean(
    "max_output_score", default=False, help="Whether to use max output score or not."
)

FLAGS = flags.FLAGS


# CONFIG ANALYSIS
TSNE = True
CONFUSION_MATRIX = True
FEATURE_MEAN_REPR = True
INSPECT_WEIGHTS = False


def do_tsne(
    known_features,
    known_labels,
    classes_dict,
    nb_classes,
    unknown_features=None,
    unknown_labels=None,
    class_anchors=None,
    mean_centers=None,
):

    # TSNE Visualization
    viz.features.tsne(
        known_features,
        known_labels,
        classes_dict.copy(),
        nb_classes,
        save_path=os.path.join(
            FLAGS.save_path, FLAGS.prefix + "tsne." + FLAGS.save_format
        ),
        class_anchors=class_anchors,
        mean_centers=mean_centers,
        title="TSNE visualization of known features",
    )

    if unknown_features is not None:
        viz.features.tsne(
            known_features,
            known_labels,
            classes_dict.copy(),
            nb_classes,
            unknown_features,
            save_path=os.path.join(
                FLAGS.save_path, FLAGS.prefix + "tsne_unknowns." + FLAGS.save_format
            ),
            class_anchors=class_anchors,
            mean_centers=mean_centers,
            title="TSNE visualization of known and unknown features",
        )

        if unknown_labels is not None:
            viz.features.tsne(
                known_features,
                known_labels,
                classes_dict.copy(),
                nb_classes,
                unknown_features,
                unknown_labels,
                save_path=os.path.join(
                    FLAGS.save_path,
                    FLAGS.prefix + "tsne_known_unknowns." + FLAGS.save_format,
                ),
                class_anchors=class_anchors if FLAGS.plot_anchors else None,
                mean_centers=mean_centers if FLAGS.actualize_centers else None,
                title="TSNE visualization of known and labelled unknown features",
            )

            viz.features.tsne(
                unknown_features,
                unknown_labels,
                classes_dict.copy(),
                nb_classes,
                save_path=os.path.join(
                    FLAGS.save_path,
                    FLAGS.prefix + "tsne_unknowns_only." + FLAGS.save_format,
                ),
                class_anchors=class_anchors if FLAGS.plot_anchors else None,
                mean_centers=mean_centers if FLAGS.actualize_centers else None,
                title="TSNE visualization of unknown features only",
            )


def get_rejection_threshold(
    known_pred_raw, unknown_pred_raw, roc_score_helper, tpr_percentage=0.9
):
    # Get OSR scores from predictions
    score_known = roc_score_helper.osr_score(known_pred_raw)
    score_unknown = roc_score_helper.osr_score(unknown_pred_raw)
    y_true, y_pred = roc_score_helper._format_score(score_known, score_unknown)

    # Compute ROC thresholds from the scores
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Search for the threshold that gives the desired TPR
    i = 0
    while tpr[i] < tpr_percentage:
        i += 1

    selected_threshold = thresholds[i] * np.sign(thresholds[i])

    print(
        f"For a TPR of {tpr[i]}, FPR is {fpr[i]} (i.e. {fpr[i]*100:.2f}% of knowns are classified as unknowns)"
    )
    print("Threshold:", selected_threshold)

    return selected_threshold


def main(argv):
    # assert FLAGS.flagfile is not None, "Please specify a flagfile."

    w = "with" if FLAGS.data_augmentation else "without"
    print(
        f"Model trained for {FLAGS.epochs} epochs {w} data augmentation on images of size {FLAGS.image_size}."
    )
    print(f"Model: {FLAGS.model}")
    print(f"Loss: {FLAGS.loss}")
    print(f"Nb features: {FLAGS.nb_features}")
    print(f"Anchor multiplier: {FLAGS.anchor_multiplier}")

    # tf.keras.utils.set_random_seed(FLAGS.seed)

    # Load dataset
    start_time = u.get_time()
    datasets, nb_classes, nb_batches, nb_channels, norm_layer, classes_dict = (
        load_dataset(
            FLAGS, shuffle=False, parallel_strategy=None, data_augmentation=False
        )
    )
    print("--- Data preprocessing time : %s ---" % (u.exec_time(start_time)))
    print(classes_dict)

    class_anchors = tf.repeat(tf.eye(nb_classes), FLAGS.nb_features, axis=1)
    class_anchors *= FLAGS.anchor_multiplier

    # Load model
    print(FLAGS.save_path, FLAGS.prefix)
    MODEL_PATH = os.path.join(FLAGS.save_path, FLAGS.prefix) + "model.save"
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    if FLAGS.nb_features > 1:
        last_conv = None
        for layer in model.layers[::-1]:
            if "conv" in layer.name:
                last_conv = layer
                break

        def gw_m(i):
            return last_conv.get_weights()[0][:, :, :, i]

        print(gw_m(0))
        print(gw_m(1))

        print(nb_classes, FLAGS.nb_features)
        print(last_conv.get_weights()[0].shape)

        print(
            "Standard deviation over all weights:",
            np.mean(
                np.std(
                    last_conv.get_weights()[0].reshape(
                        -1, FLAGS.nb_features * nb_classes
                    ),
                    axis=1,
                )
            ),
        )

        weights_per_class = None
        for i in range(nb_classes):
            w = gw_m(
                list(range(i * FLAGS.nb_features, (i + 1) * FLAGS.nb_features))
            ).reshape(-1, FLAGS.nb_features)
            total_variation = np.mean(np.std(w, axis=1))
            print(
                f"Standard deviation on the {i*FLAGS.nb_features}th to {(i+1)*FLAGS.nb_features}th set of weights : {total_variation}"
            )
            if weights_per_class is not None:
                print(
                    f"Standard deviation for set of weights {i-1} and {i} : {np.mean(np.std(np.concatenate((w, weights_per_class), axis=1), axis=1))}"
                )

            weights_per_class = w

    if INSPECT_WEIGHTS:
        viz.vizualize_weights(
            last_conv.get_weights()[0],
            0,
            FLAGS.nb_features,
            title="Weights for class 0",
            save_path=os.path.join(
                FLAGS.save_path, FLAGS.prefix + "weights." + FLAGS.save_format
            ),
        )

        viz.vizualize_weights(
            last_conv.get_weights()[0],
            1,
            FLAGS.nb_features,
            title="Weights for class 1",
            save_path=os.path.join(
                FLAGS.save_path, FLAGS.prefix + "weights." + FLAGS.save_format
            ),
        )

    # Define feature extraction
    if FLAGS.analyse == "representation":
        print("Looking at representation of the model")
        get_features = tf.keras.models.Model(
            inputs=model.inputs,
            # outputs=model.get_layer(name="global_average_pooling2d").output,
            outputs=model.get_layer(name="features_layer").output,
        )
    elif FLAGS.analyse == "output":
        print("Looking at output of the model")
        get_features = model
    else:
        raise ValueError(f"Unknown value for 'analyse': {FLAGS.analyse}")

    loss_helper = helpers.get_loss_helper(FLAGS, class_anchors, nb_classes)

    # Compute new class centers if needed
    if FLAGS.actualize_centers:
        # need to unbatch everything cause otherwise dataset isn't going to be
        # seen deterministically
        train_images, train_labels = zip(*datasets["ds_train_known"].unbatch())
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        mean_centers = u.actualize_centers(
            model,
            get_features,
            train_images,
            train_labels,
            loss_helper,
            nb_classes,
            FLAGS.batch_size,
        )
    else:
        mean_centers = class_anchors

    # Override osr_score method with max 'output' score if needed
    if FLAGS.max_output_score:

        def osr_score(y_pred, class_anchors=None):
            return tf.reduce_max(y_pred, axis=1)

        roc_score_helper = helpers.get_loss_helper(FLAGS, mean_centers, nb_classes)
        roc_score_helper.osr_score = osr_score  # override osr_score method
        roc_score_helper.score_type = "max"
    else:
        roc_score_helper = loss_helper

    #####################
    # Get images, labels and features for known and unknown images from test set
    known_images, known_labels = zip(*datasets["ds_test_known"].unbatch())
    known_images = np.array(known_images)
    known_labels = np.array(known_labels)
    label_names = [classes_dict[i] for i in range(nb_classes)]

    known_features = get_features.predict(known_images)
    known_pred_raw = model.predict(known_images)
    known_pred_raw_label = loss_helper.predicted_class(known_pred_raw)

    unknown_images, raw_unknown_labels = zip(*datasets["ds_test_unknown"].unbatch())
    unknown_images = np.array(unknown_images)
    raw_unknown_labels = np.array(raw_unknown_labels)

    unknown_features = get_features.predict(unknown_images)
    unknown_labels = np.array(
        [-label - 1 for label in raw_unknown_labels]
    )  # negated-1 labels for unknowns
    unknown_pred_raw = model.predict(datasets["ds_test_unknown"])
    unknown_pred_raw_label = loss_helper.predicted_class(unknown_pred_raw)
    #####################

    if FLAGS.plot:
        u.plot_roc_curve(
            known_pred_raw,
            unknown_pred_raw,
            roc_score_helper,
            save=os.path.join(
                FLAGS.save_path, FLAGS.prefix + "roc_curve." + FLAGS.save_format
            ),
            plot=True,
        )

    rejection_threshold = get_rejection_threshold(
        known_pred_raw, unknown_pred_raw, roc_score_helper
    )

    known_pred_thr_label = roc_score_helper.predict_w_threshold(
        known_pred_raw, threshold=rejection_threshold, type=roc_score_helper.score_type
    )
    rejected_k_preds = known_pred_thr_label == -1
    accepted_k_preds = ~rejected_k_preds

    unknown_pred_thr_label = roc_score_helper.predict_w_threshold(
        unknown_pred_raw,
        threshold=rejection_threshold,
        type=roc_score_helper.score_type,
    )
    rejected_uk_preds = unknown_pred_thr_label == -1
    accepted_uk_preds = ~rejected_uk_preds

    # ( DEBUG ckeck tpr / fpr scores
    r = np.unique(known_pred_thr_label, return_counts=True)
    print("Knowns:", r[0], r[1] / len(known_pred_thr_label))

    r = np.unique(unknown_pred_thr_label, return_counts=True)
    print("Unknowns:", r[0], r[1] / len(unknown_pred_thr_label))
    # )

    # Compute mean class features
    mean_features = viz.features.compute_mean_features(
        known_features, known_labels, nb_classes
    )
    # print("Mean features:", mean_features)

    max_features = np.max(known_features)

    viz.features.features_similarity(mean_features, label_names)

    if TSNE:
        do_tsne(
            known_features,
            known_labels,
            classes_dict,
            nb_classes,
            unknown_features,
            unknown_labels,
            class_anchors=class_anchors if FLAGS.plot_anchors else None,
            mean_centers=mean_centers if FLAGS.actualize_centers else None,
        )

    if CONFUSION_MATRIX:
        viz.confusion_matrix(
            known_labels,
            known_pred_raw_label,
            label_names,
            save_path=os.path.join(
                FLAGS.save_path, FLAGS.prefix + "confusion_matrix." + FLAGS.save_format
            ),
        )

        viz.confusion_unknown(
            unknown_labels,
            unknown_pred_raw_label,
            classes_dict,
            nb_classes,
            save_path=os.path.join(
                FLAGS.save_path,
                FLAGS.prefix + "confusion_all_unknown." + FLAGS.save_format,
            ),
        )

        viz.confusion_unknown(
            unknown_labels[accepted_uk_preds],
            unknown_pred_raw_label[accepted_uk_preds],
            classes_dict,
            nb_classes,
            save_path=os.path.join(
                FLAGS.save_path,
                FLAGS.prefix + "confusion_accepted_unknown." + FLAGS.save_format,
            ),
        )

    if FEATURE_MEAN_REPR:
        viz.features.plot_mean_features(
            mean_features,
            FLAGS.nb_features,
            label_names,
            identified_features=FLAGS.loss == "dist",
            save_path=os.path.join(
                FLAGS.save_path, FLAGS.prefix + "mean_features." + FLAGS.save_format
            ),
        )

    if FLAGS.plot:
        plt.show()


if __name__ == "__main__":
    app.run(main)
