import numpy as np
import tensorflow as tf

from helpers.base import BaseLossHelper
from losses.new_loss_v2 import loss_belong


class NewPropositionHelper2(BaseLossHelper):
    """
    Using ReLu like functions to compute the loss.
    """

    def __init__(self, nb_classes, nb_features, anchor_multiplier):
        self.nb_classes = nb_classes
        self.nb_features = nb_features
        self.anchor_val = anchor_multiplier
        self.loss_fn = loss_belong(nb_classes, nb_features, anchor_multiplier)
        self.score_type = "max"

    def loss(self, y_true, y_pred, write=False):
        return self.loss_fn(y_true, y_pred, write)

    def predicted_class(self, pred, class_anchors=None):
        x = tf.reshape(pred, (-1, self.nb_classes, self.nb_features))
        x_magnitude = tf.norm(x, axis=2)

        return tf.math.argmax(x_magnitude, axis=1)

    def osr_score(self, pred, class_anchors=None):
        x = tf.reshape(pred, (-1, self.nb_classes, self.nb_features))
        x_magnitude = tf.norm(x, axis=2)
        return tf.reduce_max(x_magnitude, axis=1)

    def _format_score(self, score_known, score_unknown):
        y_true = tf.concat(
            [np.zeros_like(score_known), np.ones_like(score_unknown)], axis=0
        )
        y_pred = tf.concat([-score_known, -score_unknown], axis=0)
        return y_true, y_pred
