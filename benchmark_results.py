import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
from pprint import pprint

import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from absl import flags
from IPython.display import display

from config_flags import load_flags
from datasets.data import load_dataset
from helpers import CrossEntropyHelper, get_loss_helper
from metrics.osr import evaluation
from utils.utils import actualize_centers

# Loading data

RES_DIR = "results/benchmark_osr/"

def list_final_dirs(directory):
    results_paths = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if "flags.txt" in f:
                results_paths.append(os.path.join(root, f))
    return results_paths


def extract_results(
    flagfile,
    calculate_cac_centers=False,
    calculate_ce_centers=False,
    calculate_softmax=False,
):
    # only one option at a time
    assert sum([calculate_cac_centers, calculate_ce_centers, calculate_softmax]) <= 1

    load_flags(flagfile)
    run_flags = flags.FLAGS

    history_path = os.path.join(run_flags.save_path, run_flags.prefix) + "history.pkl"

    if calculate_cac_centers:
        return actualize_cac_results(flagfile)
    elif calculate_ce_centers:
        return crossentropy_representations(flagfile)
    elif calculate_softmax:
        return softmax_auroc(flagfile)
    else:
        try:
            with open(history_path, "rb") as f:
                history = pickle.load(f)
        except:
            print(f"Run {flagfile} has no history.pkl, potentially crashed.")
            return None

        return {
            "accuracy": history["test_accuracy"][-1],
            "real_auroc": history["test_real_auroc"][-1],
            "max_val_auroc": history["test_max_val_auroc"][-1],
            "oscr": history["test_oscr"][-1],
        }


################################################################################
# Functions for results that have to be re-run


def actualize_cac_results(flagfile):
    load_flags(flagfile)
    args = flags.FLAGS
    args.verbose = 0

    tf.keras.utils.set_random_seed(args.seed)

    # Load dataset
    datasets, nb_classes, nb_batches, nb_channels, norm_layer, _ = load_dataset(
        args, parallel_strategy=None, data_augmentation=False
    )

    class_anchors = tf.repeat(tf.eye(nb_classes), args.nb_features, axis=1)
    class_anchors *= args.anchor_multiplier

    loss_helper = get_loss_helper(args, class_anchors, nb_classes)

    model_path = os.path.join(args.save_path, args.prefix) + "model.save"

    model = tf.keras.models.load_model(model_path)

    # need to unbatch everything cause otherwise dataset isn't going to be
    # seen deterministically
    images, labels = zip(*datasets["ds_train_known"].unbatch())
    images = np.array(images)
    labels = np.array(labels)

    preds = model.predict(images, batch_size=args.batch_size)
    predicted_label = loss_helper.predicted_class(preds)
    correct_preds_indices = predicted_label == labels
    correct_preds = preds[correct_preds_indices]
    correct_labels = labels[correct_preds_indices]

    # compute cac centers
    class_centers = []
    for i in range(nb_classes):
        class_centers.append(np.mean(correct_preds[correct_labels == i], axis=0))
    print("CAC centers computed")
    # print(class_centers)

    ##### TEST on new centers
    new_loss_helper = get_loss_helper(args, class_centers, nb_classes)

    images_test_k, labels = zip(*datasets["ds_test_known"].unbatch())
    images_test_k = np.array(images_test_k)
    labels_test_k = np.array(labels)

    pred_known = model.predict(images_test_k, batch_size=args.batch_size)
    pred_unknown = model.predict(datasets["ds_test_unknown"])

    print("Test results :")
    print("\tUsing original anchors:")
    results_old = evaluation(pred_known, pred_unknown, labels_test_k, loss_helper)
    print("\n\tUsing updated anchors:")
    results = evaluation(pred_known, pred_unknown, labels_test_k, new_loss_helper)

    return {
        "accuracy": results["acc"] * 100.0,
        "real_auroc": results["real_auroc"],
        "max_val_auroc": results["max_val_auroc"],
        "oscr": results["oscr"],
    }


def crossentropy_representations(flagfile):
    load_flags(flagfile)
    args = flags.FLAGS
    args.verbose = 0

    tf.keras.utils.set_random_seed(args.seed)

    # Load dataset
    datasets, nb_classes, nb_batches, nb_channels, norm_layer, _ = load_dataset(
        args, parallel_strategy=None, data_augmentation=False
    )

    class_anchors = tf.repeat(tf.eye(nb_classes), args.nb_features, axis=1)
    class_anchors *= args.anchor_multiplier

    loss_helper = get_loss_helper(args, class_anchors, nb_classes)

    model_path = os.path.join(args.save_path, args.prefix) + "model.save"

    model = tf.keras.models.load_model(model_path)

    features_extractor = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(name="features_layer").output,
    )

    # need to unbatch everything cause otherwise dataset isn't going to be
    # seen deterministically
    images, labels = zip(*datasets["ds_train_known"].unbatch())
    images = np.array(images)
    labels = np.array(labels)

    ce_centers = actualize_centers(
        model, features_extractor, images, labels, loss_helper, nb_classes
    )
    print("Cross entropy centers computed")
    # print(ce_centers)

    ##### TEST on new centers
    new_loss_helper = CrossEntropyHelper(osr_score="min", use_softmax=True)
    new_loss_helper.use_class_anchors(ce_centers)

    images_test_k, labels = zip(*datasets["ds_test_known"].unbatch())
    images_test_k = np.array(images_test_k)
    labels_test_k = np.array(labels)

    pred_known = features_extractor.predict(images_test_k, batch_size=args.batch_size)
    pred_unknown = features_extractor.predict(datasets["ds_test_unknown"])

    print("Test results :")
    print("\n\tUsing distance to cross entropy centers:")
    results = evaluation(pred_known, pred_unknown, labels_test_k, new_loss_helper)

    return {
        "accuracy": results["acc"],
        "real_auroc": results["real_auroc"],
        "max_val_auroc": results["max_val_auroc"],
        "oscr": results["oscr"],
    }


def softmax_auroc(flagfile):
    load_flags(flagfile)
    args = flags.FLAGS
    args.verbose = 0

    tf.keras.utils.set_random_seed(args.seed)

    # Load dataset
    datasets, nb_classes, nb_batches, nb_channels, norm_layer, _ = load_dataset(
        args, parallel_strategy=None, data_augmentation=False
    )

    class_anchors = tf.repeat(tf.eye(nb_classes), args.nb_features, axis=1)
    class_anchors *= args.anchor_multiplier

    model_path = os.path.join(args.save_path, args.prefix) + "model.save"

    model = tf.keras.models.load_model(model_path)

    new_loss_helper = CrossEntropyHelper(osr_score="max", use_softmax=True)

    images_test_k, labels = zip(*datasets["ds_test_known"].unbatch())
    images_test_k = np.array(images_test_k)
    labels_test_k = np.array(labels)

    pred_known = model.predict(images_test_k, batch_size=args.batch_size)
    pred_unknown = model.predict(datasets["ds_test_unknown"])

    print("Test results :")
    print("\n\tUsing softmax score:")
    results = evaluation(pred_known, pred_unknown, labels_test_k, new_loss_helper)

    return {
        "accuracy": results["acc"],
        "real_auroc": results["real_auroc"],
        "max_val_auroc": results["max_val_auroc"],
        "oscr": results["oscr"],
    }


def extract_data():
    res_file = "results.pkl"
    if os.path.exists(f"{RES_DIR}/{res_file}"):
        answer = input(f"File {RES_DIR}/{res_file} already exists. Overwrite? (y/n)")
        if answer.lower() != "y":
            return pickle.load(open(f"{RES_DIR}/{res_file}", "rb"))

    flagfiles = list_final_dirs(RES_DIR)
    print("\n".join(flagfiles[:5]))
    print(len(flagfiles))

    # Regroup results for different splits and re-run some models for
    # additional results
    repetition_dict = {}
    for file in flagfiles:
        config = file.rsplit("/", 1)[0].replace(RES_DIR, "")
        print(config)
        initial_loss = config.split("/")[0]
        dataset = config.split("/")[1]
        split_idx = config.split("/")[2]

        # if dataset != "tiny_imagenet":
        #     continue

        if initial_loss == "crossentropy":
            losses = [
                ("crossentropy_softmax", False, False, True),
                ("crossentropy_mos", False, False, False),
                ("crossentropy_centers", False, True, False),
            ]
        elif initial_loss == "cac":
            losses = [
                ("cac_adapted_centers", True, False, False),
                ("cac_no_adapted", False, False, False),
            ]
        else:
            losses = [(initial_loss, False, False, False)]

        for loss, *params in losses:
            results = extract_results(file, *params)
            if results is None:
                continue

            if (loss, dataset) not in repetition_dict:
                repetition_dict[(loss, dataset)] = {
                    "accuracy": [],
                    "real_auroc": [],
                    "max_val_auroc": [],
                    "oscr": [],
                }

            for key, value in results.items():
                if key == "accuracy":
                    if value < 1.0:
                        value *= 100.0
                repetition_dict[(loss, dataset)][key].append(value)

    pickle.dump(repetition_dict, open(f"{RES_DIR}/{res_file}", "wb"))

    return repetition_dict


def compute_means(repetition_dict):
    mean_results = {}
    for key, value in repetition_dict.items():
        mean_results[key] = {k: np.mean(v) for k, v in value.items()}
    return mean_results


def compute_std(repetition_dict):
    std_results = {}
    for key, value in repetition_dict.items():
        std_results[key] = {k: np.std(v) for k, v in value.items()}
    return std_results


def format_data(mean_results):
    # transform data in a datafram where columns are the datasets and rows
    # are the different losses then export it as latex table
    tmp = pd.DataFrame(mean_results)

    # each row now represent a specific table to be extracted
    for row in tmp.iterrows():
        print(row[0])
        df = (
            pd.DataFrame(row[1])
            .reset_index()
            .pivot_table(values=row[0], index="level_0", columns="level_1")
        )
        df.columns = df.columns.str.upper()
        # df = df[["MNIST", "SVHN", "CIFAR10", "CIFAR+10", "CIFAR+50", "TINY_IMAGENET"]]
        # df = df[["MNIST", "SVHN", "CIFAR10", "CIFAR+10", "CIFAR+50"]]
        # df = df[["TINY_IMAGENET"]]
        datasets = ["MNIST", "SVHN", "CIFAR10", "CIFAR+10", "CIFAR+50", "TINY_IMAGENET"]
        df = df[[d for d in datasets if d in df.columns]]

        df = df.rename_axis(None, axis=1).rename_axis(None, axis=0)
        print(df.to_latex(float_format="%.2f"))


if __name__ == "__main__":
    repetition_dict = extract_data()
    pprint(repetition_dict)
    mean_results = compute_means(repetition_dict)
    format_data(mean_results)

    std_results = compute_std(repetition_dict)
    format_data(std_results)
