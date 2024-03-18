import tensorflow as tf

from helpers.base import BaseLossHelper
from losses.compact_hypersphere import loss_compact_hyperspheres
from losses.utils import distance


class CompactHypersphereHelper(BaseLossHelper):
    def __init__(self, class_anchors, nb_classes, osr_score, m, s, _lambda, kappa):
        self.class_anchors = class_anchors
        self.loss_fn = loss_compact_hyperspheres(
            class_anchors, nb_classes, m, s, _lambda, kappa
        )
        if osr_score in ["min", "max"]:
            self.score_type = osr_score

    def loss(self, y_true, y_pred, write=False):
        return self.loss_fn(y_true, y_pred, write)

    def predicted_class(self, pred, class_anchors=None):
        if class_anchors is None:
            class_anchors = self.class_anchors
        dist = distance(pred, class_anchors)

        if self.score_type == "min":
            return tf.math.argmin(dist, axis=1)
        elif self.score_type == "max":
            return tf.math.argmax(dist, axis=1)
        else:
            raise ValueError(
                f"Unknown OSR score for compact hypersphere loss: {self.score_type}"
            )

    def osr_score(self, pred, class_anchors=None):
        if class_anchors is None:
            class_anchors = self.class_anchors
        dist = distance(pred, class_anchors)

        if self.score_type == "min":
            return tf.math.reduce_min(dist, axis=1)
        elif self.score_type == "max":
            return tf.math.reduce_max(dist, axis=1)
        else:
            raise ValueError(
                f"Unknown OSR score for compact hypersphere loss: {self.score_type}"
            )
