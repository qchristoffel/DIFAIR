import numpy as np
import tensorflow as tf


def nbf_loss_true_class(x, nb_features, features_accepted=2, swap_value=0.25, slope=1):
    """
    - features_accepted : number of features accepted for the true class
    - swap_value : loss value at 'nb_features-features_accepted'
    - slope : slope of the 'rejection' part of the function
    """
    coef = swap_value / (nb_features - features_accepted)
    return tf.maximum(
        tf.cast(-coef * x + coef * nb_features, tf.float32),
        -slope * x + slope * features_accepted + swap_value,
    )


def nbf_loss_other_class(x, features_accepted=2, swap_value=0.25, slope=1):
    """
    - features_accepted : number of features accepted for the other classes
    - swap_value : loss value at 'features_accepted'
    - slope : slope of the 'rejection' part of the function
    """
    coef = swap_value / features_accepted
    return tf.maximum(
        tf.cast(coef * x, tf.float32), slope * (x - features_accepted) + swap_value
    )


def mgn_loss_true_class(
    x,
    max_magnitude,
    magnitude_below=2,
    swap_value_below=0.25,
    slope_below=1,
    magnitude_above=1,
    swap_value_above=0.25,
    slope_above=1,
):
    """
    - magnitude_accepted : difference of magnitude accepted below and above
        'max_magnitude'fpr the true class
    - swap_value : loss value at 'max_magnitude-magnitude_accepted' and
        'max_magnitude+magnitude_accepted'
    - slope : slope of the 'rejection' part of the function
    """
    coef_below = swap_value_below / magnitude_below
    value_below = tf.where(
        x > max_magnitude - magnitude_below,
        tf.cast(-coef_below * x + coef_below * max_magnitude, tf.float32),
        -slope_below * x
        + slope_below * (max_magnitude - magnitude_below)
        + swap_value_below,
    )

    coef_above = swap_value_above / magnitude_above
    value_above = tf.where(
        x < max_magnitude + magnitude_above,
        tf.cast(coef_above * x - coef_above * max_magnitude, tf.float32),
        slope_above * (x - max_magnitude - magnitude_above) + swap_value_above,
    )
    return tf.where(x < max_magnitude, value_below, value_above)


def mgn_loss_other_class(x, magnitude_accepted=2, swap_value=0.25, slope=1):
    """
    - magnitude_accepted : maximum magnitude value accepted for the other classes
    - swap_value : loss value at 'magnitude_accepted'
    - slope : slope of the 'rejection' part of the function
    """
    coef = swap_value / magnitude_accepted
    return tf.maximum(
        tf.cast(coef * x, tf.float32), slope * (x - magnitude_accepted) + swap_value
    )


def loss_belong(nb_classes, nb_features, anchor_multiplier):
    # features_config = (
    #     {  # args nb_features is used here but the config is for nb_features=5
    #         "true_class": {
    #             "nb_features": nb_features,
    #             "features_accepted": 7,
    #             "swap_value": 0.5,
    #             "slope": 2,
    #         },
    #         "other_class": {"features_accepted": 1, "swap_value": 0.5, "slope": 2},
    #     }
    # )

    # magnitude_config = {
    #     "true_class": {
    #         "max_magnitude": np.linalg.norm([anchor_multiplier] * nb_features),
    #         "magnitude_below": 2,
    #         "swap_value_below": 0.5,
    #         "slope_below": 2,
    #         "magnitude_above": 1.5,
    #         "swap_value_above": 0.5,
    #         "slope_above": 1,
    #     },
    #     "other_class": {
    #         "magnitude_accepted": np.linalg.norm(
    #             [anchor_multiplier]
    #             * features_config["other_class"]["features_accepted"]
    #         ),
    #         "swap_value": 1,
    #         "slope": 2,
    #     },
    # }

    features_config = {
        # args nb_features is used here but the config is for nb_features=5
        "true_class": {
            "nb_features": nb_features,
            "features_accepted": 4,
            "swap_value": 0.5,
            "slope": 2,
        },
        "other_class": {"features_accepted": 1, "swap_value": 0.5, "slope": 2},
    }

    magnitude_config = {
        "true_class": {
            "max_magnitude": np.linalg.norm([anchor_multiplier] * nb_features),
            "magnitude_below": 2,
            "swap_value_below": 0.5,
            "slope_below": 2,
            "magnitude_above": 1.5,
            "swap_value_above": 0.5,
            "slope_above": 1,
        },
        "other_class": {
            "magnitude_accepted": np.linalg.norm(
                [anchor_multiplier]
                * features_config["other_class"]["features_accepted"]
            ),
            "swap_value": 1,
            "slope": 2,
        },
    }

    nbf_true_class_fn = lambda x: nbf_loss_true_class(
        x, **features_config["true_class"]
    )
    nbf_other_class_fn = lambda x: nbf_loss_other_class(
        x, **features_config["other_class"]
    )

    mgn_true_class_fn = lambda x: mgn_loss_true_class(
        x, **magnitude_config["true_class"]
    )
    mgn_other_class_fn = lambda x: mgn_loss_other_class(
        x, **magnitude_config["other_class"]
    )

    @tf.function
    def _loss(y_true, x, write=False):
        """
        x : extracted features, array of shape (batch_size, nb_features*nb_classes)
        labels : true class
        nb_features : number of features
        """
        x = tf.reshape(x, (-1, nb_classes, nb_features))
        # 0.5 * anchor multiplier is the threshold to consider a feature as activated
        x_count = tf.reduce_sum(
            tf.where(x > (0.5 * anchor_multiplier), 1.0, 0.0), axis=2
        )

        x_magnitude = tf.norm(x, axis=2)

        indices = tf.stack(
            [tf.range(tf.shape(y_true)[0], dtype=tf.int64), y_true], axis=1
        )

        # ici on accepte plus ou moins le nombre de features activées par classe
        tmp = nbf_true_class_fn(tf.gather_nd(x_count, indices))
        x_count = nbf_other_class_fn(x_count)

        # x_count = tf.tensor_scatter_nd_update(x_count, indices, tmp)

        x_count = tf.tensor_scatter_nd_update(x_count, indices, tf.zeros_like(tmp))

        loss_count = tmp + tf.reduce_max(x_count, axis=1)
        # loss_count = tmp + tf.reduce_sum(x_count, axis=1) / (nb_classes-1)

        # on veut que la magnitude de la vraie classe soit supérieure aux autres
        tmp = mgn_true_class_fn(tf.gather_nd(x_magnitude, indices))
        x_magnitude = mgn_other_class_fn(x_magnitude)
        # x_magnitude = tf.tensor_scatter_nd_update(x_magnitude, indices, tmp)

        x_magnitude = tf.tensor_scatter_nd_update(
            x_magnitude, indices, tf.zeros_like(tmp)
        )
        loss_magnitude = tmp + tf.reduce_max(x_magnitude, axis=1)
        # loss_magnitude = tmp + tf.reduce_sum(x_magnitude, axis=1) / (nb_classes-1)

        if write:
            tf.summary.scalar("losses/loss_count", tf.reduce_mean(loss_count))
            tf.summary.scalar("losses/loss_magnitude", tf.reduce_mean(loss_magnitude))

        # return tf.reduce_sum(x_count, axis=1) + tf.reduce_sum(x_magnitude, axis=1)
        # return 3 * loss_count + loss_magnitude
        return loss_magnitude / loss_count
        # return (loss_count / 10) + (loss_magnitude / 10)

    return _loss
