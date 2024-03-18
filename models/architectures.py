import tensorflow as tf
from tensorflow import keras

layers = keras.layers

from functools import partial

from losses.losses import (
    different_weights_loss_corr,
    different_weights_loss_std,
    exponential_penalty,
    linear_penalty,
)
from metrics.metrics import weights_convergence_metric
from utils.utils import exec_time, get_time


def get_model(args, nb_classes, nb_channels, norm_layer):
    input_shape = (args.image_size, args.image_size, nb_channels)

    model_args = {
        "input_shape": input_shape,
        "nb_classes": nb_classes,
        "nb_features": args.nb_features,
        "normalization": norm_layer,
        "fc_end": args.fc_end,
        "reconstruction": args.reconstruction,
        "last_conv_activation": args.last_conv_activation,
        "last_conv_bn": args.last_conv_bn,
    }

    if args.reconstruction and args.model != "standard_vgg32":
        raise NotImplementedError(
            "Reconstruction is not implemented yet for architectures other than standard_vgg32."
        )

    if args.last_conv_activation != "none" and args.model != "standard_vgg32":
        raise NotImplementedError(
            """Using specific activation after last layer is not implemented yet for architectures other than standard_vgg32."""
        )

    if args.model == "standard_vgg32":
        model = create_VGG32(**model_args)
    elif args.model == "6_layers":
        model = create_6_layers(**model_args)
    elif args.model == "vgg16":
        model = create_VGG16(**model_args)
    elif args.model == "resnet50":
        model = create_ResNet50(**model_args)
    elif args.model == "efb0":
        model = create_EfficientNetB0(**model_args)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    return model


def wrap_model(
    model,
    input_shape,
    nb_classes,
    nb_features,
    normalization=None,
    fc_end=False,
    last_conv_activation="none",
    last_conv_bn=False,
):
    inputs = keras.Input(shape=input_shape)

    # Rescale here (augmentation need data between 0 and 255)
    # Like that, test data and new data will be rescaled and normalized as part of the model
    out = keras.layers.Rescaling(1.0 / 255)(inputs)

    if normalization:
        out = normalization(out)

    out = model(out)

    # Now finish the model

    # If the model outputs a representation
    if not fc_end:
        # Add a convolutional layer that will have as much filters
        # as the number of features of the representation
        last_conv = layers.Conv2D(
            filters=nb_features * nb_classes,
            kernel_size=3,
            use_bias=False,
            padding="same",
            name="last_conv",
        )

        match last_conv_activation:
            case "relu":
                act = layers.ReLU()
            case "leaky_relu":
                act = layers.LeakyReLU(0.2)
            case "sigmoid":
                act = layers.Activation("sigmoid")
            case _:
                act = layers.Activation("linear")

        if last_conv_bn:
            bn = layers.BatchNormalization()

            out = act(bn(last_conv(out)))
        else:
            out = act(last_conv(out))

        representation = layers.GlobalAveragePooling2D(name="features_layer")(out)

        outputs = [representation]

    elif fc_end and nb_features > 1:
        # if fc_end and a specific number of features is given,
        # add a conv layer which will output the right number of features
        last_conv = layers.Conv2D(
            filters=nb_features,
            kernel_size=3,
            use_bias=False,
            padding="same",
            name="last_conv",
        )

        act = layers.LeakyReLU(0.2)

        if last_conv_bn:
            bn = layers.BatchNormalization()

            out = act(bn(last_conv(out)))
        else:
            out = act(last_conv(out))

        representation = layers.GlobalAveragePooling2D(name="features_layer")(out)

        outputs = [layers.Dense(nb_classes, use_bias=False)(representation)]
    else:
        representation = layers.GlobalAveragePooling2D(name="features_layer")(out)

        # if fc_end and nb_features <= 1, add a dense layer
        outputs = [layers.Dense(nb_classes, use_bias=False)(representation)]

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def conv_block(
    inputs,
    filters,
    activation="leaky_relu",
    padding="same",
    bias=False,
    initializer="he_normal",  # he_uniform
    dropout=0.2,
):
    out = inputs

    if dropout:
        out = layers.Dropout(dropout)(out)

    initializer_conv = partial(
        tf.keras.initializers.RandomNormal, mean=0.0, stddev=0.05
    )

    for i, f in enumerate(filters):
        out = layers.Conv2D(
            f,
            3,
            padding=padding,
            strides=1 if i != len(filters) - 1 else 2,
            kernel_initializer=initializer_conv(),
            use_bias=bias,
        )(out)
        out = layers.BatchNormalization()(out)

        if activation == "leaky_relu":
            out = layers.LeakyReLU(0.2)(out)
        else:
            out = layers.Activation(activation)(out)

    return out


def create_VGG32(
    input_shape,
    nb_classes,
    nb_features,
    normalization=None,
    fc_end=False,
    reconstruction=False,
    last_conv_activation="none",
    last_conv_bn=False,
):
    # Architecture configuration
    architecture = [[64, 64, 128], [128, 128, 128], [128, 128, 128]]
    dropout = [0.2, 0.2, 0.2]

    inputs = keras.Input(shape=input_shape)

    # Rescale now (augmentation need data between 0 and 255)
    # Like that, test data and new data will be rescaled and normalized as part of the model
    out = keras.layers.Rescaling(1.0 / 255)(inputs)

    # Normalize (this layer is generated and 'adapted' when loading data)
    if normalization:
        out = normalization(out)

    # Generate main architecture
    for filters, d in zip(architecture, dropout):
        out = conv_block(out, filters=filters, dropout=d)

    # After conv blocks, finish the model

    # If the model outputs a representation
    if not fc_end:
        # Add a convolutional layer that will have as much filters
        # as the number of features of the representation
        last_conv = layers.Conv2D(
            filters=nb_features * nb_classes,
            kernel_size=3,
            use_bias=False,
            padding="same",
            name="last_conv",
        )

        match last_conv_activation:
            case "relu":
                act = layers.ReLU()
            case "leaky_relu":
                act = layers.LeakyReLU(0.2)
            case "sigmoid":
                act = layers.Activation("sigmoid")
            case _:
                act = layers.Activation("linear")

        if last_conv_bn:
            bn = layers.BatchNormalization()

            out = act(bn(last_conv(out)))
        else:
            out = act(last_conv(out))

        conv_to_reconstruct = out

        representation = layers.GlobalAveragePooling2D(name="features_layer")(out)

        reshaped_repr = tf.keras.layers.Reshape((1, 1, nb_features * nb_classes))(
            representation
        )

        if reconstruction:
            reconstruction_0 = layers.Conv2DTranspose(
                filters=nb_features * nb_classes,
                kernel_size=3,
                strides=4,
                padding="same",
                name="reconstruction_0",
            )

            match last_conv_activation:
                case "relu":
                    act_0 = layers.ReLU()
                case "leaky_relu":
                    act_0 = layers.LeakyReLU(0.2)
                case "sigmoid":
                    act_0 = layers.Activation("sigmoid")
                case _:
                    act_0 = layers.Activation("linear")

            # reconstruction_1 = layers.Conv2DTranspose(
            #     filters=128,
            #     kernel_size=3,
            #     strides=1,
            #     padding="same",
            #     name="reconstruction_1",
            # )

            # act_1 = layers.LeakyReLU(0.2)

            out = act_0(reconstruction_0(reshaped_repr))
            # out = act_1(reconstruction_1(out))

            reconstruction_output = out

            outputs = [representation, conv_to_reconstruct, reconstruction_output]
        else:
            outputs = [representation]

    elif fc_end and nb_features > 1:
        # if fc_end and a specific number of features is given,
        # add a conv layer which will output the right number of features
        last_conv = layers.Conv2D(
            filters=nb_features,
            kernel_size=3,
            use_bias=False,
            padding="same",
            name="last_conv",
        )

        act = layers.LeakyReLU(0.2)

        if last_conv_bn:
            bn = layers.BatchNormalization()

            out = act(bn(last_conv(out)))
        else:
            out = act(last_conv(out))

        representation = layers.GlobalAveragePooling2D(name="features_layer")(out)

        outputs = [layers.Dense(nb_classes, use_bias=False)(representation)]
    else:
        representation = layers.GlobalAveragePooling2D(name="features_layer")(out)

        # if fc_end and nb_features <= 1, add a dense layer
        outputs = [layers.Dense(nb_classes, use_bias=False)(representation)]

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def create_6_layers(
    input_shape,
    nb_classes,
    nb_features,
    normalization=None,
    fc_end=False,
    add_layer_loss=False,
):
    architecture = [[64, 128], [128, 128], [128, 128]]
    dropout = [0.2, 0.2, 0.2]

    inputs = keras.Input(shape=input_shape)

    # Rescale here (augmentation need data between 0 and 255)
    # Like that, test data and new data will be rescaled and normalized as part of the model
    out = keras.layers.Rescaling(1.0 / 255)(inputs)

    if normalization:
        out = normalization(out)

    for filters, d in zip(architecture, dropout):
        out = conv_block(out, filters=filters, dropout=d)

    if not fc_end:
        features_layer = layers.Conv2D(
            filters=nb_features * nb_classes,
            kernel_size=3,
            use_bias=False,
            padding="same",
            name="last_conv",
        )
        # features_layer.add_loss(different_weights_loss(features_layer.get_weights()[0], nb_classes))
        out = features_layer(out)

    outputs = layers.GlobalAveragePooling2D(name="features_layer")(out)

    if fc_end:
        outputs = layers.Dense(nb_classes, use_bias=False)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def create_VGG16(
    input_shape,
    nb_classes,
    nb_features,
    normalization=None,
    fc_end=False,
    add_layer_loss=True,
):
    vgg16 = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
    )

    return wrap_model(
        vgg16,
        input_shape,
        nb_classes,
        nb_features,
        normalization=normalization,
        fc_end=fc_end,
        add_layer_loss=add_layer_loss,
    )


def create_ResNet50(
    input_shape,
    nb_classes,
    nb_features,
    normalization=None,
    fc_end=False,
    add_layer_loss=True,
):
    resnet50 = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
    )

    return wrap_model(
        resnet50,
        input_shape,
        nb_classes,
        nb_features,
        normalization=normalization,
        fc_end=fc_end,
        add_layer_loss=add_layer_loss,
    )


def create_EfficientNetB0(
    input_shape,
    nb_classes,
    nb_features,
    normalization=None,
    fc_end=False,
    add_layer_loss=True,
):
    efb0 = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
    )

    return wrap_model(
        efb0,
        input_shape,
        nb_classes,
        nb_features,
        normalization=normalization,
        fc_end=fc_end,
        add_layer_loss=add_layer_loss,
    )


# def conv_block(inputs,
#                filters,
#                activation="leaky_relu",
#                padding="same",
#                initializer="he_normal", # he_uniform
#                batch_norm=True,
#                dropout=0.2,
#                pooling=2,
#                weight_decay=1e-4,
#                filter_size=3):

#     regularizer = None
#     if weight_decay:
#         regularizer = keras.regularizers.l2(weight_decay)

#     out = layers.Conv2D(filters, filter_size,
#                         activation=activation,
#                         padding=padding,
#                         kernel_initializer=initializer,
#                         kernel_regularizer=regularizer)(inputs)
#     if batch_norm : out = layers.BatchNormalization()(out)
#     out = layers.Conv2D(filters, filter_size,
#                         activation=activation,
#                         padding=padding,
#                         kernel_initializer=initializer,
#                         kernel_regularizer=regularizer)(out)
#     if batch_norm : out = layers.BatchNormalization()(out)

#     if pooling:
#         out = layers.MaxPool2D(pooling)(out)
#     if dropout:
#         out = layers.Dropout(dropout)(out)

#     return out

# def create_model(input_shape, nb_classes, nb_features, normalization=None, softmax=False):
#     inputs = keras.Input(shape=input_shape)

#     architecture = [32,64,128]
#     dropout = [0.2, 0.3, 0.4]

#     out = inputs
#     for filters, d in zip(architecture, dropout):
#         out = conv_block(out, filters=filters, dropout=d)

#     features_layer = layers.Conv2D(filters=nb_features*nb_classes,
#                         kernel_size=1,
#                         padding="same")
#     out = features_layer(out)

#     features = layers.GlobalAveragePooling2D(name="features_layer")(out)
#     # outputs = layers.Dense(output_size)(outputs)

#     model = keras.Model(inputs=inputs, outputs=features)

#     def different_weights_loss(weights):
#         def _custom_loss():
#             loss = tf.math.reduce_mean(
#                 tf.math.reduce_std(
#                     weights.reshape((-1, nb_classes, nb_features)),
#                     axis=2
#                 ),
#                 axis=0
#             )

#             loss = 2*tf.math.exp(-2*loss)
#             loss = tf.math.reduce_sum(loss)
#             return loss
#         return _custom_loss

#     model.add_loss(different_weights_loss(features_layer.get_weights()[0]))

#     return model


def predict_distance(model, ds, anchors):
    """Compute the distance between the predictions from the model on the dataset
    and the anchors.
    """

    preds = model.predict(ds)

    dist = tf.norm(tf.expand_dims(preds, 1) - anchors, axis=2)

    return dist
