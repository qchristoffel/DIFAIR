import tensorflow as tf

from helpers.base import BaseLossHelper
from losses.difair import hard_loss, smooth_loss
from losses.losses import reconstruction_loss
from losses.utils import distance


class DifairHelper(BaseLossHelper):
    def __init__(
        self,
        class_anchors,
        max_dist,
        nb_classes,
        nb_features,
        difair_type="hard",
        osr_score="min",
        reconstruction_weight=0,
    ):
        self.class_anchors = class_anchors
        self.nb_classes = nb_classes
        self.nb_features = nb_features
        self.difair_type = difair_type
        self._reconstruction = reconstruction_weight > 0

        if osr_score in ["min", "max"]:
            self.score_type = osr_score
        else:
            raise ValueError(f"Unknown OSR score for distance loss: {osr_score}")

        if self.difair_type == "hard":
            self.difair_loss = hard_loss(class_anchors, max_dist)
        elif self.difair_type == "smooth":
            self.difair_loss = smooth_loss(class_anchors, max_dist)
        else:
            raise ValueError(f"Unknown DIFAIR loss type: {self.difair_type}")

        if self._reconstruction:
            self.reconstruction_loss = reconstruction_loss(reconstruction_weight)

            def combined_loss(y_true, y_pred):
                return self.difair_loss(y_true, y_pred[0]) + self.reconstruction_loss(
                    y_pred[1], y_pred[2]
                )

            self.loss_fn = combined_loss
        else:
            self.loss_fn = self.difair_loss

    @property
    def distance_based(self):
        return True

    @property
    def reconstruction(self):
        return self._reconstruction

    def loss(self, y_true, y_pred, write=False):
        return self.loss_fn(y_true, y_pred)

    def predicted_class(self, pred, class_anchors=None, threshold=None):
        if class_anchors is None:
            class_anchors = self.class_anchors
        dist = distance(pred, class_anchors)

        if self.score_type == "min":
            preds = tf.math.argmin(dist, axis=1)
        elif self.score_type == "max":
            preds = tf.math.argmax(dist, axis=1)
        else:
            raise ValueError(f"Unknown OSR score for distance loss: {self.score_type}")

        return preds

    def prediction_score(self, pred, class_anchors=None):
        if class_anchors is None:
            class_anchors = self.class_anchors
        dist = distance(pred, class_anchors)

        return dist

    def osr_score(self, pred, class_anchors=None):
        if class_anchors is None:
            class_anchors = self.class_anchors
        dist = distance(pred, class_anchors)

        if self.score_type == "min":
            return tf.math.reduce_min(dist, axis=1)
        elif self.score_type == "max":
            return tf.math.reduce_max(dist, axis=1)
        else:
            raise ValueError(f"Unknown OSR score for distance loss: {self.score_type}")

    # def auroc_v2(self, pred_known, pred_unknown):
    #     nb_classes = 6
    #     nb_features = 5
    #     reshaped_pred_k = tf.reshape(pred_known, (-1, nb_classes, nb_features))
    #     reshaped_pred_u = tf.reshape(pred_unknown, (-1, nb_classes, nb_features))
    #     score_known = tf.reduce_max(tf.reduce_sum(reshaped_pred_k, axis=2), axis=1)
    #     score_unknown = tf.reduce_max(tf.reduce_sum(reshaped_pred_u, axis=2), axis=1)
    #     y_true = tf.concat([np.zeros_like(score_known), np.ones_like(score_unknown)], axis=0)
    #     score_known, score_unknown = modify_score(score_known, score_unknown, "max")
    #     y_pred = tf.concat([score_known, score_unknown], axis=0)
    #     return roc_auc_score(y_true, y_pred)
