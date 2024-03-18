import tensorflow as tf
from absl import flags
from tensorflow import keras

FLAGS = flags.FLAGS

from utils.utils import tf_correlation


def weights_convergence_metric(weights, nb_classes, nb_features):
    metric = tf.math.reduce_mean(
        tf.math.reduce_std(
            tf.reshape(weights, (-1, nb_classes, nb_features)),
            axis=2,
        ),
        axis=0,
    )

    return metric


def weights_correlation_metric(weights, nb_classes, nb_features):
    # Version using the mean of the whole upper triangle
    # correlations = tf_correlation(
    #     tf.reshape(weights, (-1, nb_classes * nb_features)),
    # )

    # metric_value = tf.math.reduce_mean(
    #     tf.abs(
    #         tf.linalg.band_part(correlations, 0, -1)
    #         - tf.linalg.band_part(correlations, 0, 0)
    #     )
    # )

    # Version using the mean of the upper triangle of each class
    correlations = tf_correlation(tf.reshape(weights, (-1, nb_classes * nb_features)))
    correlations -= tf.linalg.band_part(correlations, 0, 0)

    metric_value = 0.0
    for i in range(nb_classes):
        sub_matrix = correlations[
            i * nb_features : (i + 1) * nb_features,
            i * nb_features : (i + 1) * nb_features,
        ]
        values = tf.experimental.numpy.triu(sub_matrix, 1)
        metric_value += tf.math.reduce_mean(tf.abs(values))

    return metric_value


class AccuracyAllMetrics(keras.metrics.Metric):
    def __init__(self, anchors, nb_classes, name="accuracy_all_metrics", **kwargs):
        super(AccuracyAllMetrics, self).__init__(name=name, **kwargs)
        self.class_anchors = anchors
        self.nb_classes = nb_classes

        self.accuracy = tf.keras.metrics.Accuracy()
        self.accuracy_cosine = tf.keras.metrics.Accuracy()
        self.accuracy_dist = tf.keras.metrics.Accuracy()

    @tf.function
    def compute_metrics(self, y_pred):
        dot_product = tf.matmul(y_pred, tf.transpose(self.class_anchors))
        norm_y_pred = tf.norm(y_pred, axis=1, keepdims=True)
        norm_anchors = tf.norm(self.class_anchors, axis=1)

        # cos_sim = dot_product / (norm_y_pred * tf.transpose(norm_anchors))
        cos_sim = dot_product * 1 / tf.transpose(norm_anchors)
        cos_sim = cos_sim * 1 / norm_y_pred

        # broadcasting subtraction
        dist = tf.norm(tf.expand_dims(y_pred, 1) - self.class_anchors, axis=2)

        return cos_sim, dist

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        sim, dist = self.compute_metrics(y_pred)
        combination = tf.multiply(1 - sim, dist)  # 1-sim cause sim is positive
        # if sim was 1 then the combination is minimal because of the
        # multiplication by 0 or if dist is 0 it is minimal too. Thing is it
        # would maybe be better to have a distance of 0 which means that the
        # point is exactly on the anchor rather than a similarity of 0 which
        # means that the vector is in the same direction.

        self.accuracy.update_state(y_true, tf.argmin(combination, axis=1))
        self.accuracy_cosine.update_state(y_true, tf.argmax(sim, axis=1))
        self.accuracy_dist.update_state(y_true, tf.argmin(dist, axis=1))

    def result(self):
        return {
            "acc_combined": self.accuracy.result(),
            "acc_cosine": self.accuracy_cosine.result(),
            "acc_dist": self.accuracy_dist.result(),
        }

    def reset_state(self):
        self.accuracy.reset_states()
        self.accuracy_cosine.reset_states()
        self.accuracy_dist.reset_states()


class Accuracy_CAC(keras.metrics.Metric):
    """Accuracy without using threshold to reject classes."""

    def __init__(self, anchors, nb_classes, name="accuracy_cac", **kwargs):
        super(Accuracy_CAC, self).__init__(name=name, **kwargs)
        self.class_anchors = anchors
        self.nb_classes = nb_classes

        self.accuracy = tf.keras.metrics.Accuracy()

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        dist = tf.norm(tf.expand_dims(y_pred, 1) - self.class_anchors, axis=2)

        gamma = get_rejection_scores(dist)

        self.accuracy.update_state(y_true, tf.argmin(gamma, axis=1))
