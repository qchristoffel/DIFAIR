import numpy as np
import tensorflow as tf

from helpers.base import BaseLossHelper
from losses.cac import loss_cac
from losses.utils import distance, softmin


class CACHelper(BaseLossHelper):
    def __init__(self, class_anchors, nb_classes, nb_features):
        self.class_anchors = class_anchors
        self.loss_fn = loss_cac(class_anchors)
        self.score_type = "min"
        self.nb_classes = nb_classes
        self.nb_features = nb_features

    def loss(self, y_true, y_pred, write=False):
        return self.loss_fn(y_true, y_pred, write)

    def predicted_class(self, pred, class_anchors=None):
        """Return the predicted class for each prediction."""

        if class_anchors is None:
            class_anchors = self.class_anchors
        # compute distance to all anchors
        dist = distance(pred, class_anchors)
        gamma = self._get_rejection_scores(dist)

        # TODO: take in account rejection of a prediction
        return tf.math.argmin(gamma, axis=1)

    def _get_rejection_scores(self, dist):
        return dist * (1 - softmin(dist))

    def osr_score(self, pred, class_anchors=None):
        """Compute OSR scores, this is one value per prediction."""

        if class_anchors is None:
            class_anchors = self.class_anchors
        # compute distance to all anchors
        dist = distance(pred, class_anchors)

        # compute rejection scores
        gamma = self._get_rejection_scores(dist)

        return tf.reduce_min(gamma, axis=1)

    def _format_score(self, score_known, score_unknown):
        y_true = tf.concat(
            [np.zeros_like(score_known), np.ones_like(score_unknown)], axis=0
        )
        y_pred = tf.concat([score_known, score_unknown], axis=0)
        return y_true, y_pred
