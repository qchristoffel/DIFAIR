################################################################################
# This script is used to run a detailed search of hyperparameters for a given
# configuration of losses used and model. The hyperparameters are the learning
# rate, the number of features, the anchor multiplier and the distance ratio,
# and the usage of correlation loss or not.
################################################################################

import itertools
import os
import subprocess
from pprint import pprint

import numpy as np

################################################################################
# Define the parameters

SAVE_DIR = "results/hyperparameters_search_tinyImagenet/"
REPEATS = 5
SLURM = False

EPOCHS = 600
BATCH_SIZE = 128

standard_flags = (
    f"--epochs {EPOCHS} --batch_size {BATCH_SIZE} "
    "--verbose 2 "
    "--loss difair "
    "--difair_loss_type smooth "
    "--last_conv_activation relu "
    "--correlation_penalty "
    "--nolast_conv_bn "
    "--noreconstruction "
    "--split_train_val "
    "--summary "
    # "--dataset automobile_truck_vs_tiger "
    # "--config 0 "
)
# TODO: remove last two flags, are just there for testing on smaller dataset

# possible parameters are described in the form (<param_value>, <path_name>)
# with `path_name' beeing used to create folders of results
param_space = {
    "lr": [(0.3, "lr_0.3"), (0.4, "lr_0.4"), (0.5, "lr_0.5")],
    "nb_features": [(5, "nbf_5"), (10, "nbf_10"), (20, "nbf_20")],
    "anchor_multiplier": [(2, "am_2"), (5, "am_5"), (10, "am_10")],
    "distance_ratio": [
        (0.2, "dr_0.2"),
        (0.3, "dr_0.3"),
        (0.4, "dr_0.4"),
        (0.5, "dr_0.5"),
        (0.6, "dr_0.6"),
    ],
    # use a nested dict because combinations are exclusives
    # (i.e there is no value and weight if penalty is none )
    # "correlation_penalty": [
    #     (False, "without_corr"),
    #     {
    #         "correlation_penalty": [(True, "with_corr")],
    #         # "correlation_penalty_weight": [0.5, 1, 2],
    #     },
    # ],
}


################################################################################
# Generate all possible combinations of parameters


def product_dict(**kwargs):
    # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


# First extend the dictionary with the values of the nested dictionaries
for key, values in param_space.items():
    values_to_add = []
    for i, val in enumerate(values):
        if isinstance(val, dict):
            param_space[key].pop(i)
            values_to_add.extend(list(product_dict(**val)))
    param_space[key].extend(values_to_add)

pprint(param_space)


# Generate all possible combinations of parameters
combinations = list(product_dict(**param_space))
print(f"Total number of combinations: {len(combinations)}")


def get_flag(flag, param_value):
    if isinstance(param_value, bool):
        flag = f"--{flag if param_value else 'no' + flag} "
    else:
        flag = f"--{flag} {param_value} "
    return flag


# Launch experiments
for c in combinations:
    # Generate the flags and path to save the results
    path = SAVE_DIR
    flags = ""
    print(f"Combination: {c}")

    anchor_to_anchor = np.sqrt(c["anchor_multiplier"][0] ** 2 * c["nb_features"][0] * 2)
    max_dist = np.round(anchor_to_anchor * c["distance_ratio"][0], 3)

    flags += f"--max_dist {max_dist} "

    for flag, val in c.items():
        if flag == "distance_ratio":
            path = os.path.join(path, val[1])
            continue

        if isinstance(val, dict):
            # do not use parent key flag if nested dict
            for flag_2, val_2 in val.items():
                param_value = val_2[0]
                param_name = val_2[1]
                flags += get_flag(flag_2, param_value)

                # separate by '-' when nested, supposed to be linked parameters
                path += f"{param_name}-"
            path = os.path.join(path[:-1], "")
        else:
            param_value = val[0]
            param_name = val[1]
            flags += get_flag(flag, param_value)
            path = os.path.join(path, param_name, "")

    flags = flags.strip()
    print(f"Flags: {flags}", flush=True)
    print(f"Path: {path}", flush=True)

    os.makedirs(path, exist_ok=True)

    for i in range(REPEATS):
        print(f"\tRepeat {i+1}/{REPEATS}", flush=True)

        tensorboard = (
            "--tensorboard tsb_logs/hyperparameters_search/"
            f"{path.replace(SAVE_DIR, '')}/run_{i} "
        )

        print(f"Tensorboard: {tensorboard}", flush=True)

        if not SLURM:
            subprocess.run(
                [
                    "bash",
                    "bash_scripts/launch.sh",
                    "local",
                    "--flagfile default_flags/difair_cifar10_flags.txt",
                    f"--save_path {path}",
                    f"--prefix run_{i}_",
                    standard_flags,
                    tensorboard,
                    flags,
                ],
            )
        else:
            subprocess.run(
                [
                    "sbatch",
                    f"--output={path}/run_{i}_log_%j.out",
                    "bash_scripts/launch.sh",
                    "slurm",
                    "--flagfile default_flags/difair_tiny_imagenet_flags.txt",
                    f"--save_path {path}",
                    f"--prefix run_{i}_",
                    standard_flags,
                    tensorboard,
                    flags,
                ],
            )

    print("\n")
