import os

from absl import flags

flags.DEFINE_string(
    "model",
    default="standard_vgg32",
    help="""Model to use. Can be standard_vgg32, vgg16, resnet50, efb0.""",
)

flags.DEFINE_string(
    "dataset",
    default="mnist",
    help="Dataset to use. Can be mnist, svhn, cifar10, cifar+10, cifar+50, tiny_imagenet.",
)
flags.DEFINE_integer(
    "config", default=0, help="Select which class split configuration to use. 0 to 4."
)

flags.DEFINE_bool(
    "summary",
    default=False,
    help="If true, only print the summary of the model and don't run.",
)

flags.DEFINE_integer(
    "image_size",
    default=32,
    help="Size of the input images. Can be used to resize the images.",
)

flags.DEFINE_float(
    "lr",
    default=0.1,
    help="Learning rate. (might not be used if there is a LR schedule)",
)

flags.DEFINE_integer("seed", default=0, help="Seed for reproducibility.")

# scheduler parameters
flags.DEFINE_string(
    "scheduler", default="cosine_restart_warmup", help="Learning rate scheduler to use."
)

flags.DEFINE_integer(
    "epoch_swap",
    default=50,
    help="Epoch at which to swap the learning rate for CAC scheduler.",
)

flags.DEFINE_integer(
    "num_restarts",
    default=2,
    help="Number of restarts for the cosine restart scheduler.",
)

flags.DEFINE_integer("epochs", default=5, help="Number of epochs.")

flags.DEFINE_integer("batch_size", default=64, help="Batch size.")

flags.DEFINE_bool(
    "data_augmentation", default=True, help="If True, use data augmentation."
)

flags.DEFINE_bool(
    "split_train_val",
    default=False,
    help="If True, split the training set into a training and validation set.",
)

flags.DEFINE_bool(
    "plot", default=False, help="If true, plot the history of the training."
)

flags.DEFINE_multi_enum(
    "to_save",
    short_name="ts",
    default="all",
    enum_values=["all", "model", "history", "none"],
    help="What to save (model, history, all, none).",
)

flags.DEFINE_integer("verbose", default=1, help="Verbosity level (0, 1, 2).")

flags.DEFINE_string(
    "save_path",
    default=None,
    help="Directory to put the saves (model, history, flags).",
)

flags.DEFINE_string(
    "prefix",
    default="",
    help="""Prefix for the files to save 
                    (model, history, flags). Only used if save_path is not None.""",
)

flags.DEFINE_string(
    "tensorboard",
    default=None,
    help="""Directory to write tensorboard logs. If None, tensorboard logs will not be written.""",
)

# anchor parameters
flags.DEFINE_float(
    "anchor_multiplier",
    default=4,
    help="""Multiplier for the anchor coordinates. Default 
anchors are on the unit (hyper)sphere but the multiplier 
can set them on a (hyper)sphere with a different radius.""",
)

flags.DEFINE_integer(
    "nb_features", default=5, help="""Number of features per class in the anchor."""
)

###############################
# Loss function parameters

flags.DEFINE_string("loss", default="crossentropy", help="""Loss function to use.""")
# TODO: add list of available loss functions

flags.DEFINE_enum(
    "last_conv_activation",
    default="none",
    enum_values=["none", "relu", "leaky_relu", "sigmoid"],
    help="""Activation function to use on the last convolutional layer, before pooling.""",
)

flags.DEFINE_boolean(
    "last_conv_bn",
    default=False,
    help="""If True, add a batch normalization layer after the last convolutional layer, before activation.""",
)


# Standard deviation penalty
flags.DEFINE_enum(
    "std_penalty",
    default="none",
    enum_values=["none", "exponential", "linear"],
    help="""Standard deviation penalty used to avoid convergence of weights on the last conv.""",
)

flags.DEFINE_float(
    "std_penalty_value",
    default=1,
    help="""If std_penalty is not none the standard deviation of weights will be optimized to be
close to 'std_penalty_value' value.""",
)

flags.DEFINE_float(
    "std_penalty_weight",
    default=1,
    help="""Weight of the standard deviation penalty in the loss function.""",
)


# Correlation penalty
flags.DEFINE_boolean(
    "correlation_penalty",
    default=False,
    help="""If True, add a correlation penalty to the loss function.""",
)

flags.DEFINE_float(
    "correlation_penalty_weight",
    default=1,
    help="""Weight of the correlation penalty in the loss function.""",
)

flags.DEFINE_boolean(
    "reconstruction",
    default=False,
    help="""If True, add a reconstruction loss to the loss function. 
                    The output of the layer before the representation is reconstructed.""",
)

flags.DEFINE_float(
    "reconstruction_weight",
    default=1,
    help="""Weight of the reconstruction loss in the loss function.""",
)

###############################
# DIFAIR parameters
flags.DEFINE_float(
    "max_dist",
    default=3,
    help="""Maximum distance allowed between a class anchor and
                   representations learned for this class.""",
)

# deprecated ?
flags.DEFINE_float(
    "max_angle",
    default=35,
    help="""Maximum angle (in degree) allowed between a class 
                   anchor and representations learned for this class. 
                   CAUTION: this value should not exceed 45 degrees as anchors
                   are orthogonals (this would allow for an overlap).""",
)

flags.DEFINE_enum(
    "difair_loss_type",
    default="hard",
    enum_values=["hard", "smooth"],
    help="""Type of DIFAIR loss to use. 'Hard' is the original loss, where no 
penalty is applied if the representation is within the hypersphere.
'Smooth' is a smoothed version of the loss, where a small penalty is applied
if the representation is within the hypersphere to push it towards the center.""",
)

###############################

flags.DEFINE_string(
    "osr_score",
    default="max",
    help="""Can be max or min. If max, the maximum OSR score is 
                    taken to represent known data (for example when using maximum 
                    logit score). If min, the minimum OSR score is taken to 
                    represent known data (for example when using distance to a point).""",
)

flags.DEFINE_boolean(
    "use_softmax",
    default=False,
    help="""If True, use softmax score for crossentropy instead of logits score.""",
)

flags.DEFINE_boolean(
    "fc_end",
    default=False,
    help="""If True, add a fully connected layer after global average pooling. 
                     Otherwise, a convolution layer is added before the global average pooling 
                     to reduce the number of filters.""",
)

flags.DEFINE_integer(
    "noise_data",
    default=0,
    help="""If 'noise' > 0, add 'noise' randomly generated images to the training set to try to learn that no 
                    representation should be learned for this data.""",
)

###############################
# rand augmentation parameters
flags.DEFINE_integer(
    "randaug_n", default=1, help="""Rand Augment: Number of transformations to apply."""
)

flags.DEFINE_integer(
    "randaug_m", default=6, help="""Rand Augment: Magnitude of the transformations."""
)


def load_flags(file):
    FLAGS = flags.FLAGS

    # need to import the files to have the specific flags
    # import baseline

    # initialize flags
    _ = FLAGS(["", f"--flagfile={file}"], known_only=False)
