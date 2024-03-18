################################################################################
# This script is used to run all experiments on Neal et al. 2018 OSR benchmark.
################################################################################

import itertools
import os
import subprocess
from pprint import pprint

import numpy as np

################################################################################
# Define the parameters

SAVE_DIR = "results/benchmark_osr/"
REPEATS = 1
SLURM = False

EPOCHS = 600
BATCH_SIZE = 128
NB_FEATURES = 5
ANCHOR_MULTIPLIER = 10
DISTANCE_RATIO = 0.6

anchor_to_anchor = np.sqrt(ANCHOR_MULTIPLIER**2 * NB_FEATURES * 2)
max_dist = np.round(anchor_to_anchor * DISTANCE_RATIO, 3)

standard_flags = (
    f"--epochs {EPOCHS} --batch_size {BATCH_SIZE} "
    "--model standard_vgg32 "
    "--verbose 2 "
    "--loss difair "
    "--difair_loss_type smooth "
    f"--max_dist {max_dist} "
    f"--nb_features {NB_FEATURES} "
    f"--anchor_multiplier {ANCHOR_MULTIPLIER} "
    "--last_conv_activation relu "
    "--correlation_penalty "
    "--nolast_conv_bn "
    "--noreconstruction "
    "--nosplit_train_val "
    "--summary "
)

# Define the search space

# possible parameters are described in the form (<param_value>, <path_name>)
# with `path_name' beeing used to create folders of results

datasets = ["mnist", "svhn", "cifar10", "cifar+10", "cifar+50", "tiny_imagenet"]
losses = ["crossentropy", "difair", "cac"]

param_space = {
    "loss": list(zip(losses, losses)),
    "dataset": list(zip(datasets, datasets)),
    "config": list(zip(range(5), [f"split_{i}" for i in range(5)])),
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


# use default flags files for datasets so that
# data augmentation and lr is configured for each dataset


# Launch experiments
for c in combinations:
    # Generate the flags and path to save the results
    path = SAVE_DIR
    flags = ""
    print(f"Combination: {c}")

    for flag, val in c.items():
        if isinstance(val, dict):
            # do not use parent key flag if nested dict
            for flag_2, val_2 in val.items():
                param_value = val_2[0]
                param_name = val_2[1]
                flags += get_flag(flag_2, param_value)

                # separate by '-' when nested, supposed to be linked parameters
                path += f"{param_name}-"
            path = os.path.join(path[:-1], "")
        elif flag == "loss":
            param_value = val[0]
            param_name = val[1]
            path = os.path.join(path, param_name, "")
            if param_value == "crossentropy":
                flags += (
                    f"--loss crossentropy "
                    f"--osr_score max "
                    f"--fc_end "
                    f"--nb_features 0 "
                )
            elif param_value == "cac":
                flags += (
                    f"--loss cac "
                    f"--fc_end "
                    f"--lr 0.1 "
                    f"--nb_features 1 "
                    f"--anchor_multiplier 10 "
                    f"--nocorrelation_penalty "
                )
            else:
                flags += get_flag(flag, param_value)
                path = os.path.join(path, param_name, "")
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
                    f"--flagfile default_flags/difair_{c['dataset'][1]}_flags.txt",
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
                    f"--flagfile default_flags/difair_{c['dataset'][1]}_flags.txt",
                    f"--save_path {path}",
                    f"--prefix run_{i}_",
                    standard_flags,
                    tensorboard,
                    flags,
                ],
            )

    print("\n")
