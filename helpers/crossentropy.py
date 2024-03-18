import tensorflow as tf

from helpers.base import BaseLossHelper
from losses.utils import distance


class CrossEntropyHelper(BaseLossHelper):
    def __init__(self, osr_score="max", use_softmax=False):
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.use_softmax = use_softmax
        if osr_score in ["min", "max"]:
            self.score_type = osr_score
        else:
            raise ValueError(f"Unknown OSR score for cross entropy loss: {osr_score}")
        self.class_anchors = None

    def use_class_anchors(self, class_anchors):
        self.class_anchors = class_anchors
        self.score_type = "min"

    def loss(self, y_true, y_pred, write=False):
        return self.loss_fn(y_true, y_pred)

    def predicted_class(self, y_pred, class_anchors=None, threshold=None):
        if class_anchors is None:
            class_anchors = self.class_anchors

        if class_anchors is None:
            # if there is no anchor, y_pred is supposed to be logits
            preds = tf.argmax(y_pred, axis=1)
        else:
            # if there is an anchor, y_pred is the representation in the anchors space
            dist = distance(y_pred, class_anchors)
            preds = tf.argmin(dist, axis=1)

        return preds

    def osr_score(self, y_pred, class_anchors=None):
        if self.class_anchors is None:
            # if there is no anchor, y_pred is supposed to be logits
            if self.use_softmax:
                y_pred = tf.nn.softmax(y_pred)
            return tf.math.reduce_max(y_pred, axis=1)
        else:
            # if there is an anchor, y_pred is the representation in the anchors space
            dist = distance(y_pred, self.class_anchors)
            if self.score_type == "min":
                return tf.math.reduce_min(dist, axis=1)
            elif self.score_type == "max":
                return tf.math.reduce_max(dist, axis=1)
            else:
                raise ValueError(
                    f"Unknown OSR score for cross entropy loss: {self.score_type}"
                )
