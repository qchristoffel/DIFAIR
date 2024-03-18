import json

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from datasets.datasets_config import datasets_config
from datasets.randaugment.randaugment import RandAugmentLayer
from datasets.splits.osr_splits import osr_splits

tf.keras.utils.set_random_seed(0)

from tensorflow import keras

layers = keras.layers

AUTOTUNE = tf.data.AUTOTUNE


def get_unknown_classes(known_classes, nb_classes):
    return [i for i in range(nb_classes) if i not in known_classes]


def get_splits_info(dataset_name, config):
    splits_info = {}
    if dataset_name in ["cifar+10", "cifar+50"]:
        splits_info["unknown"] = osr_splits[dataset_name]["splits"][config]
        splits_info["known"] = [0, 1, 8, 9]  # 4 vehicle classes from cifar10
    elif "_vs_" in dataset_name:  # specific situation where unknown are specified
        splits_info["known"] = osr_splits[dataset_name]["splits"][config]
        splits_info["unknown"] = osr_splits[dataset_name]["unknown"][config]
    else:
        splits_info["known"] = osr_splits[dataset_name]["splits"][config]
        splits_info["unknown"] = get_unknown_classes(
            splits_info["known"], datasets_config[dataset_name]["nb_classes"]
        )

    if config < len(osr_splits[dataset_name]["means"]):
        splits_info["mean"] = osr_splits[dataset_name]["means"][config]
    else:
        splits_info["mean"] = None
    if config < len(osr_splits[dataset_name]["variances"]):
        splits_info["variance"] = osr_splits[dataset_name]["variances"][config]
    else:
        splits_info["variance"] = None

    return splits_info


def map_dict(d, f, ignore_keys=[]):
    new_d = {}
    for k, v in d.items():
        if v is not None and k not in ignore_keys:
            new_d[k] = v.map(f)
        else:
            new_d[k] = v
    return new_d


def get_equal_len_datasets(ds1, ds2, ds1_size=None, ds2_size=None):
    if ds1_size is None:
        ds1_size = tf.cast(ds1.reduce(0, lambda x, _: x + 1), tf.int64)
    if ds2_size is None:
        ds2_size = tf.cast(ds2.reduce(0, lambda x, _: x + 1), tf.int64)

    if ds1_size > ds2_size:
        ds1 = ds1.shuffle(ds1_size)
        ds1 = ds1.take(ds2_size)
    elif ds2_size > ds1_size:
        ds2 = ds2.shuffle(ds2_size)
        ds2 = ds2.take(ds1_size)

    return ds1, ds2


def get_fake_dataset(nb_samples, image_size=32, channels=3):
    # fake dataset for debugging
    ds_train = tf.data.Dataset.from_tensor_slices(
        (
            tf.random.uniform(
                (nb_samples, image_size, image_size, channels),
                minval=0,
                maxval=255,
                dtype=tf.dtypes.int32,
            ),
            tf.random.uniform(
                (nb_samples,), minval=0, maxval=10, dtype=tf.dtypes.int32
            ),
        )
    )
    ds_test = tf.data.Dataset.from_tensor_slices(
        (
            tf.random.uniform(
                (int(nb_samples / 2), image_size, image_size, channels),
                minval=0,
                maxval=255,
                dtype=tf.dtypes.int32,
            ),
            tf.random.uniform(
                (int(nb_samples / 2),), minval=0, maxval=10, dtype=tf.dtypes.int32
            ),
        )
    )

    class FakeInfo:
        def __init__(self):
            self.features = tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(
                        shape=(image_size, image_size, channels)
                    ),
                    "label": tfds.features.ClassLabel(num_classes=10),
                }
            )

    ds_info = FakeInfo()

    return (ds_train, ds_test), ds_info


def dataset_filtering(raw_datasets, known_classes, unknown_classes):
    @tf.function
    def filter_known(x, y):
        return tf.reduce_any(y == known_classes)

    @tf.function
    def filter_unknown(x, y):
        return tf.reduce_any(y == unknown_classes)

    @tf.function
    def reset_labels_known(x, y):
        # for an unknown reason, using tf.squeeze instead of [0, 0] doesn't
        # work in distributed training and distribute slices without dimensions,
        # whereas here it still creates empty slices but they have dimensions
        return x, tf.where(y == known_classes)[0, 0]

    # Filter datasets to keep only known or unknown classes
    # Train
    ds_train_known = raw_datasets["ds_train_known"].filter(filter_known)

    # Val
    if len(unknown_classes) != 0 and raw_datasets["ds_val_unknown"] is not None:
        ds_val_unknown = raw_datasets["ds_val_unknown"].filter(filter_unknown)
    else:
        ds_val_unknown = None

    # Test
    ds_test_known = raw_datasets["ds_test_known"].filter(filter_known)
    if len(unknown_classes) != 0:
        ds_test_unknown = raw_datasets["ds_test_unknown"].filter(filter_unknown)
    else:
        ds_test_unknown = None

    datasets = {
        "ds_train_known": ds_train_known,
        "ds_val_unknown": ds_val_unknown,
        "ds_test_known": ds_test_known,
        "ds_test_unknown": ds_test_unknown,
    }

    # Reset labels of known datasets
    datasets = map_dict(
        datasets,
        reset_labels_known,
        ignore_keys=["ds_test_unknown", "ds_val_unknown"],
    )

    return datasets


def add_noise_data(
    ds,
    nb_classes,
    image_size,
    channels,
    nb_samples=1000,
    seed=0,
):
    # Concatenate noise data to train dataset
    noise_data = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(  # cast to uint8
                tf.random.uniform(
                    (nb_samples, image_size, image_size, channels),
                    minval=0,
                    maxval=255,
                    dtype=tf.dtypes.int32,
                    seed=seed,
                ),
                tf.uint8,
            ),
            tf.constant([nb_classes] * nb_samples, dtype=tf.int64),
        )
    )

    return ds.concatenate(noise_data)


def load_dataset(
    args,
    shuffle=True,
    parallel_strategy=None,
    data_augmentation=True,
    seed=0,
):
    # set seed
    tf.random.set_seed(seed)
    # tf.keras.utils.set_random_seed(seed)

    # everything is slow/unoptimized with dataset filtering and splitting but should work

    # Load datasets, get dataset info (class splits, channels, etc.)
    if args.dataset == "fake":
        train_dataset_info = datasets_config["fake"]
        class_splits_info = {
            "known": list(range(6)),
            "unknown": list(range(6, 10)),
            "mean": None,
            "variance": None,
        }

        (ds_train, ds_test), ds_info = get_fake_dataset(
            1000, image_size=args.image_size, channels=train_dataset_info["channels"]
        )

        classes_dict = {
            new: ds_info.features["label"].int2str(old)
            for new, old in enumerate(class_splits_info["known"])
        }
        classes_dict = {
            **classes_dict,
            **{
                -i: ds_info.features["label"].int2str(i)
                for i in class_splits_info["unknown"]
            },
        }

        raw_datasets = {
            "ds_train_known": ds_train,
            "ds_test_known": ds_test,
            "ds_test_unknown": ds_test,
        }

    elif args.dataset in ["cifar+10", "cifar+50"] or "_vs_" in args.dataset:
        train_dataset_info = datasets_config["cifar10"]
        class_splits_info = get_splits_info(args.dataset, args.config)

        (ds_train_known, ds_test_known), ds_info = tfds.load(
            "cifar10",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

        (ds_train_unknown, ds_test_unknown), ds_uk_info = tfds.load(
            "cifar100",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

        # set labels for known classes (index is reset)
        classes_dict = {
            new: ds_info.features["label"].int2str(old)
            for new, old in enumerate(class_splits_info["known"])
        }
        # set labels for unknown classes using (-label)-1 (to shift label 0 to -1)
        classes_dict = {
            **classes_dict,
            **{
                -i - 1: ds_uk_info.features["label"].int2str(i)
                for i in class_splits_info["unknown"]
            },
        }

        raw_datasets = {
            # train/val dataset with known classes (cifar10)
            "ds_train_known": ds_train_known,
            # val dataset with unknown classes (cifar100)
            "ds_val_unknown": ds_train_unknown if args.split_train_val else None,
            # test dataset with known classes (cifar10)
            "ds_test_known": ds_test_known,
            # test dataset with unknown classes (cifar100)
            "ds_test_unknown": ds_test_unknown,
        }

    else:
        train_dataset_info = datasets_config[args.dataset]
        class_splits_info = get_splits_info(args.dataset, args.config)

        (ds_train, ds_test), ds_info = tfds.load(
            train_dataset_info["real_name"],
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

        # set labels for known classes (index is reset)
        classes_dict = {
            new: ds_info.features["label"].int2str(old)
            for new, old in enumerate(class_splits_info["known"])
        }
        # set labels for unknown classes using (-label)-1 (to shift label 0 to -1)
        classes_dict = {
            **classes_dict,
            **{
                -i - 1: ds_info.features["label"].int2str(i)
                for i in class_splits_info["unknown"]
            },
        }

        raw_datasets = {
            "ds_train_known": ds_train,
            "ds_val_unknown": ds_train if args.split_train_val else None,
            "ds_test_known": ds_test,
            "ds_test_unknown": ds_test,
        }

    # Filter datasets to keep only known or unknown classes
    datasets = dataset_filtering(
        raw_datasets,
        class_splits_info["known"],
        class_splits_info["unknown"],
    )

    # Add noise data to train dataset
    if args.noise_data > 0:
        datasets["ds_train_known"] = add_noise_data(
            datasets["ds_train_known"],
            len(class_splits_info["known"]),
            image_size=args.image_size,
            channels=train_dataset_info["channels"],
            nb_samples=args.noise_data,
        )

    # Split train into train and val if needed
    if args.split_train_val:
        val_split = train_dataset_info["split"]
        ds_train_known, ds_val_known = tf.keras.utils.split_dataset(
            datasets["ds_train_known"],
            right_size=val_split,
            shuffle=True,
            seed=seed,
        )

        datasets["ds_train_known"] = ds_train_known
        datasets["ds_val_known"] = ds_val_known
    else:
        datasets["ds_val_known"] = None

    # Balance test and val datasets
    if len(class_splits_info["unknown"]) != 0:
        # Equalize number of samples in known and unknown val datasets
        if args.split_train_val:
            ds_val_known, ds_val_unknown = get_equal_len_datasets(
                datasets["ds_val_known"], datasets["ds_val_unknown"]
            )
        else:
            ds_val_known, ds_val_unknown = None, None

        # Equalize number of samples in known and unknown test datasets
        ds_test_known, ds_test_unknown = get_equal_len_datasets(
            datasets["ds_test_known"], datasets["ds_test_unknown"]
        )

        ###DEBUG
        # ds_test_k_size = ds_test_known.reduce(0, lambda x, _: x + 1)
        # ds_test_uk_size = ds_test_unknown.reduce(0, lambda x, _: x + 1)
        # tf.print("ds_test_k_size:", ds_test_k_size)
        # tf.print("ds_test_uk_size:", ds_test_uk_size)
        ###DEBUG
    else:
        ds_val_unknown = None
        ds_test_known = datasets["ds_test_known"]
        ds_test_unknown = None

    datasets["ds_val_known"] = ds_val_known
    datasets["ds_val_unknown"] = ds_val_unknown
    datasets["ds_test_known"] = ds_test_known
    datasets["ds_test_unknown"] = ds_test_unknown

    train_size = (
        datasets["ds_train_known"].prefetch(AUTOTUNE).reduce(0, lambda x, _: x + 1)
    )

    print("Dataset:", args.dataset)
    print("Known classes:", class_splits_info["known"])
    print("Unknown classes:", class_splits_info["unknown"])

    tf.print("nb_train_examples:", train_size)
    nb_batches = int(np.ceil(train_size / args.batch_size))

    # Get mean and variance for this split
    mean, variance = class_splits_info["mean"], class_splits_info["variance"]
    print("mean:", mean)
    print("variance:", variance)
    if not mean or not variance:
        # Compute mean and variance for this split
        norm_layer = layers.Normalization(axis=-1)
        norm_layer.adapt(
            datasets["ds_train_known"].map(lambda x, y: x / 255), batch_size=256
        )
        print(
            "Train set mean and variance:",
            norm_layer.mean.numpy(),
            norm_layer.variance.numpy(),
        )
        print(
            "Add them to 'datasets/splits/osr_splits.py' to avoid future computation."
        )
    else:
        # Use precomputed mean and variance
        norm_layer = layers.Normalization(mean=mean, variance=variance, axis=-1)

    # Normalization model
    resize_and_rescale = tf.keras.Sequential(
        [
            layers.Resizing(args.image_size, args.image_size),
            # layers.Rescaling(1./255), # don't rescale RandAugment need values between [0,255]
        ]
    )

    # Augmentation model
    if args.data_augmentation and data_augmentation:
        if args.dataset in ["svhn"]:
            exclude_ops = ["Rotate", "TranslateX", "TranslateY"]
        else:
            exclude_ops = None

        da_model = data_augmentation_model(
            input_shape=(
                args.image_size,
                args.image_size,
                train_dataset_info["channels"],
            ),
            args=args,
            flip="horizontal" if args.dataset not in ["mnist", "svhn"] else None,
            exclude_ops=exclude_ops,
        )
    else:
        da_model = None

    # Prepare datasets
    datasets["ds_train_known"] = prepare(
        datasets["ds_train_known"],
        preprocess=resize_and_rescale,
        augment=da_model,
        shuffle=shuffle,
        batch_size=args.batch_size,
    )
    datasets = {
        k: (
            v
            if k == "ds_train_known" or v is None
            else prepare(v, preprocess=resize_and_rescale, batch_size=args.batch_size)
        )
        for k, v in datasets.items()
    }

    # Distribute datasets
    if parallel_strategy:
        datasets = {
            k: parallel_strategy.experimental_distribute_dataset(v)
            for k, v in datasets.items()
        }

    return (
        datasets,
        len(class_splits_info["known"]),
        nb_batches,
        train_dataset_info["channels"],
        norm_layer,
        classes_dict,
    )


def data_augmentation_model(input_shape, args, flip="horizontal", exclude_ops=None):
    input = x = layers.Input(shape=input_shape)
    if args.dataset == "mnist":  # only one channel so RandAugment doesn't work
        x = layers.RandomContrast(0.5)(x)
        x = layers.RandomBrightness(0.5)(x)
    else:
        print("Using rand augment")
        x = RandAugmentLayer(
            n=args.randaug_n, m=args.randaug_m, exclude_ops=exclude_ops
        )(x)

    # reproduce RandomCrop transform behavior from "A good closed set model is all you need"
    x = layers.ZeroPadding2D(padding=4)(x)
    x = layers.RandomCrop(args.image_size, args.image_size)(x)

    if flip:
        x = layers.RandomFlip(flip)(x)

    return tf.keras.Model(inputs=input, outputs=x, name="data_augmentation_model")


def prepare(
    ds, preprocess, augment=None, shuffle=False, shuffle_size=2000, batch_size=128
):
    """
    'preprocess' = function / keras model that will be mapped to every data
    'augment' = keras model with preprocessing layers to augment data
    """

    if preprocess:
        # Preprocess dataset (generaly resize and rescale)
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)

    # Cache dataset (but need to go through whole dataset when iterating or create bugs)
    ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(shuffle_size)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE
        )

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)
