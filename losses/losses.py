import numpy as np
import tensorflow as tf

from utils.utils import tf_correlation


def exponential_penalty(alpha, tau, epsilon):
    # loss function is: alpha * exp(-beta * std)
    # this loss function is equal to epsilon when std = tau
    beta = tf.math.log(alpha / epsilon) / tau

    def _f(std):
        return alpha * tf.math.exp(-beta * std)

    return _f


def linear_penalty(slope, intercept):
    # loss function is (notice the slope is negated already): -slope * std + slope * intercept
    # this loss function is equal to 0 when std = intercept

    def _f(std):
        return -slope * std + slope * intercept

    return _f


def different_weights_loss_std(
    weights, nb_classes, nb_features, penalty=None, loss_weight=1.0
):
    if penalty is None:
        penalty = exponential_penalty(
            alpha=2,
            tau=1,
            epsilon=0.02,
        )
        # penalty = linear_penalty(slope=2, intercept=1)

    @tf.function
    def _custom_loss():
        # penalize the std of the weights
        loss = penalty(
            tf.math.reduce_std(
                tf.reshape(weights, (-1, nb_classes, nb_features)), axis=2
            )
        )

        loss = tf.math.reduce_mean(loss) * loss_weight

        return loss

    return _custom_loss


def different_weights_loss_corr(weights, nb_classes, nb_features, loss_weight=1.0):
    # Version using the mean of the whole upper triangle
    # @tf.function
    # def _custom_loss():
    #     correlations = tf_correlation(
    #         tf.reshape(weights, (-1, nb_classes * nb_features))
    #     )

    #     # minimize the mean correlation between weights (no distinction between classes here)
    #     loss = tf.math.reduce_mean(
    #         tf.abs(
    #             tf.linalg.band_part(correlations, 0, -1)
    #             - tf.linalg.band_part(correlations, 0, 0)
    #         )
    #     )

    #     return loss * loss_weight

    # Version using the mean of the upper triangle of each class
    @tf.function
    def _custom_loss():
        correlations = tf_correlation(
            tf.reshape(weights, (-1, nb_classes * nb_features))
        )
        correlations -= tf.linalg.band_part(correlations, 0, 0)

        loss = 0.0
        for i in range(nb_classes):
            sub_matrix = correlations[
                i * nb_features : (i + 1) * nb_features,
                i * nb_features : (i + 1) * nb_features,
            ]
            values = tf.experimental.numpy.triu(sub_matrix, 1)
            loss += tf.math.reduce_mean(tf.abs(values))

        loss *= loss_weight
        # loss = (loss / nb_classes) * loss_weight

        return loss

    return _custom_loss


def add_last_conv_penalties(last_conv, nb_classes, args):
    weights_penalties_fn = {}

    # no need to add loss to maximize std if nb_features = 1 since it's always 0
    if args.nb_features > 1 and args.std_penalty != "none":
        if args.std_penalty == "linear":
            penalty = linear_penalty(slope=2, intercept=args.std_penalty_value)

            weights_penalties_fn["std"] = different_weights_loss_std(
                last_conv.trainable_variables[0],
                nb_classes,
                args.nb_features,
                penalty=penalty,
                loss_weight=args.std_penalty_weight,
            )
            last_conv.add_loss(weights_penalties_fn["std"])

        elif args.std_penalty == "exponential":
            penalty = exponential_penalty(
                alpha=2, tau=args.std_penalty_value, epsilon=1e-3
            )

            weights_penalties_fn["std"] = different_weights_loss_std(
                last_conv.trainable_variables[0],
                nb_classes,
                args.nb_features,
                penalty=penalty,
                loss_weight=args.std_penalty_weight,
            )

            last_conv.add_loss(weights_penalties_fn["std"])
        else:
            raise ValueError(f"Unknown std_penalty: {args.std_penalty}")

    if args.correlation_penalty:
        weights_penalties_fn["corr"] = different_weights_loss_corr(
            last_conv.trainable_variables[0],
            nb_classes,
            args.nb_features,
            loss_weight=args.correlation_penalty_weight,
        )

        last_conv.add_loss(weights_penalties_fn["corr"])

    return weights_penalties_fn


def reconstruction_loss(reconstruction_weight=1.0):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def _loss(y_true, y_pred):
        # sometimes there are empty batches distributed to replicas
        batch_size = tf.shape(y_true)[0]
        if batch_size == 0:
            return tf.reshape(tf.constant(0, dtype=tf.float32), (-1,))

        ## Reconstruction loss
        reconstruction_loss = (
            mse(
                tf.reshape(y_true, (batch_size, -1)),
                tf.reshape(y_pred, (batch_size, -1)),
            )
            * reconstruction_weight
        )

        tf.summary.scalar("loss/reconstruction", tf.reduce_mean(reconstruction_loss))

        return reconstruction_loss

    return _loss


def loss_sim_dist(class_anchors, max_angle, max_dist):
    @tf.function
    def _loss(y_true, y_pred, write=False):
        # sometimes there are empty batches distributed to replicas
        batch_size = tf.shape(y_true)[0]
        if batch_size == 0:
            return tf.reshape(tf.constant(0, dtype=tf.float32), (-1,))

        # y true is not one hot encoded
        representation_to_learn = tf.gather(class_anchors, y_true)

        sim = tf.keras.losses.cosine_similarity(representation_to_learn, y_pred, axis=1)
        dist = tf.norm(y_pred - representation_to_learn, axis=1)

        relax_sim = tf.maximum(sim + max_angle, 0)
        relax_dist = tf.maximum(dist - max_dist, 0)

        # relax_sim = tf.math.exp(relax_sim)-1
        # relax_dist = tf.math.square(relax_dist)-1

        return relax_sim + relax_dist

    return _loss


def penalize_wrong_classification(class_anchors, max_dist):
    @tf.function
    def _loss(y_true, y_pred):
        # sometimes there are empty batches distributed to replicas
        batch_size = tf.shape(y_true)[0]
        if batch_size == 0:
            return tf.reshape(tf.constant(0, dtype=tf.float32), (-1,))

        # y true is not one hot encoded
        representation_to_learn = tf.gather(class_anchors, y_true)

        dist = tf.norm(y_pred - representation_to_learn, axis=1)
        distance_to_all_anchors = tf.norm(
            tf.expand_dims(y_pred, 1) - class_anchors, axis=2
        )

        relax_dist = tf.maximum(dist - max_dist, 0)

        # penalize wrong classification
        relax_dist = tf.where(
            tf.argmin(distance_to_all_anchors, axis=1) == y_true,
            relax_dist,
            relax_dist**2,
        )

        dist_difference = tf.expand_dims(dist, axis=1) - distance_to_all_anchors
        difference_sums = tf.reduce_sum(
            tf.where(
                tf.math.not_equal(dist_difference, 0), tf.math.exp(dist_difference), 0
            ),
            axis=1,
        )

        tuplet_loss = tf.math.log(1 + difference_sums)

        # relax_sim = tf.math.exp(relax_sim)-1
        # relax_dist = tf.math.square(relax_dist)-1

        return relax_dist + tuplet_loss

    return _loss


def loss_individual_dimensions(class_anchors, nb_classes, nb_features, max_dist):
    @tf.function
    def _loss(y_true, y_pred):
        # sometimes there are empty batches distributed to replicas
        batch_size = tf.shape(y_true)[0]
        if batch_size == 0:
            return tf.reshape(tf.constant(0, dtype=tf.float32), (-1,))

        representation_to_learn = tf.gather(class_anchors, y_true)
        representation_to_learn = tf.reshape(
            representation_to_learn, (-1, nb_classes, nb_features)
        )
        reshaped_pred = tf.reshape(y_pred, (-1, nb_classes, nb_features))

        distances = tf.norm(reshaped_pred - representation_to_learn, axis=2)

        if y_true[0] == 0:
            tf.print(distances[0][:10], summarize=-1)
            tf.print("y_true", y_true[0], "\n", summarize=-1)
            tf.print("y_pred", y_pred[0], summarize=-1)

        # distance to class center (on class dimensions)
        dist_to_self = tf.gather_nd(
            distances,
            tf.stack(
                [tf.range(tf.shape(distances)[0], dtype=tf.int64), y_true], axis=1
            ),
        )

        # distance to the origin (on other class dimensions)
        dist_to_others = tf.reduce_sum(distances, axis=1) - dist_to_self

        relax_dist = tf.maximum(dist_to_self - max_dist, 0)

        # TODO : pondérer en fonction de la "présence" des features ? 5 features sur
        # un vecteur de 50 face à 45 autre features

        # return 2*relax_dist + dist_to_others - 0.01*tf.reduce_sum(y_pred, axis=1)
        return (nb_classes) * relax_dist + dist_to_others

    return _loss
