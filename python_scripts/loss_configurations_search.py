################################################################################
# This script is used to run a detailed search of the loss configuration
# namely using specific penalties for weights of the last conv layer, using
# activations and batch norm after last convolutional layer.
################################################################################

import itertools
import os
import subprocess
from pprint import pprint

import numpy as np

################################################################################
# Define the parameters

SAVE_DIR = "results/loss_configurations_search"
REPEATS = 5
SLURM = False

EPOCHS = 600
BATCH_SIZE = 128
NB_FEATURES = 20
ANCHOR_MULTIPLIER = 2

anchor_to_anchor = np.sqrt(ANCHOR_MULTIPLIER**2 * NB_FEATURES * 2)
DISTANCE_RATIO = 0.4
MAX_DIST = np.round(anchor_to_anchor * DISTANCE_RATIO, 3)

standard_flags = (
    f"--epochs {EPOCHS} --batch_size {BATCH_SIZE} "
    f"--nb_features {NB_FEATURES} "
    f"--anchor_multiplier {ANCHOR_MULTIPLIER} --max_dist {MAX_DIST} "
    "--verbose 2 "
    "--loss difair "
    "--noreconstruction "
    "--summary "
    # "--dataset automobile_truck_vs_tiger "
    # "--config 0 "
)
# TODO: remove last two flags, are just there for testing on smaller dataset


# Define the search space

# possible parameters are described in the form (<param_value>, <path_name>)
# with `path_name' beeing used to create folders of results
param_space = {
    "last_conv_bn": [(True, "with_bn"), (False, "without_bn")],
    # "last_conv_activation": [("none", "without_act"), ("relu", "with_act")],
    "last_conv_activation": [("leaky_relu", "with_act_leaky")],
    # use a nested dict because combinations are exclusives
    # (i.e there is no value and weight if penalty is none )
    "std_penalty": [
        ("none", "without_std"),
        {
            "std_penalty": [("exponential", "std_exp"), ("linear", "std_linear")],
            "std_penalty_value": [(0.1, "0.1"), (0.2, "0.2")],
            # "std_penalty_weight": [0.5, 1, 2],
        },
    ],
    "correlation_penalty": [
        (False, "without_corr"),
        {
            "correlation_penalty": [(True, "with_corr")],
            # "correlation_penalty_weight": [0.5, 1, 2],
        },
    ],
    "difair_loss_type": [("hard", "hard_loss"), ("smooth", "smooth_loss")],
    # "reconstruction": [
    #     (False, "without_rec"),
    #     {
    #         "reconstruction": [(True, "with_rec")],
    #         # "reconstruction_weight": [1],
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
    # print(f"Combination: {c}")

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
        else:
            param_value = val[0]
            param_name = val[1]
            flags += get_flag(flag, param_value)
            path = os.path.join(path, param_name, "")

    flags += (
        "--tensorboard tsb_logs/loss_configurations_search/"
        f"{path.replace(SAVE_DIR, '')} "
    )

    flags = flags.strip()
    print(f"Flags: {flags}", flush=True)
    print(f"Path: {path}", flush=True)

    os.makedirs(path, exist_ok=True)

    for i in range(REPEATS):
        print(f"\tRepeat {i+1}/{REPEATS}", flush=True)
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
                    "--flagfile default_flags/difair_cifar10_flags.txt",
                    f"--save_path {path}",
                    f"--prefix run_{i}_",
                    standard_flags,
                    flags,
                ],
            )

    print("\n")
