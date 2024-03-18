import abc

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve

from helpers.utils import modify_score


class BaseLossHelper(abc.ABC):
    def __init__(self):
        pass

    @property
    def distance_based(self):
        return False

    @property
    def reconstruction(self):
        return False

    @abc.abstractmethod
    def loss(self, y_true, y_pred, write=False):
        pass

    @abc.abstractmethod
    def predicted_class(self, y_pred, class_anchors=None):
        # return index of predicted class
        pass

    @abc.abstractmethod
    def osr_score(self, y_pred, class_anchors=None):
        """Compute OSR scores, this is one value per prediction."""

        # return osr score for prediction
        # y_pred is a tensor of shape (batch_size, nb_features)
        pass

    def prediction_score(self, y_pred, class_anchors=None):
        return y_pred

    def predict_w_threshold(self, y_pred, threshold, type="min", class_anchors=None):
        """Return the predicted class for each prediction."""
        score = self.osr_score(y_pred, class_anchors)
        preds = self.predicted_class(y_pred, class_anchors)

        if type == "min":
            return tf.where(score < threshold, preds, -1)
        elif type == "max":
            return tf.where(score > threshold, preds, -1)
        else:
            raise ValueError(f"Unknown type for threshold: {type}")

    def _format_score(self, score_known, score_unknown):
        y_true = tf.concat(
            [np.zeros_like(score_known), np.ones_like(score_unknown)], axis=0
        )
        score_known, score_unknown = modify_score(
            score_known, score_unknown, self.score_type
        )
        y_pred = tf.concat([score_known, score_unknown], axis=0)
        return y_true, y_pred

    def auroc(self, pred_known, pred_unknown):
        score_known = self.osr_score(pred_known)
        score_unknown = self.osr_score(pred_unknown)
        y_true, y_pred = self._format_score(score_known, score_unknown)
        return roc_auc_score(y_true, y_pred)

    def plot_roc_curve(self, pred_known, pred_unknown):
        score_known = self.osr_score(pred_known)
        score_unknown = self.osr_score(pred_unknown)
        y_true, y_pred = self._format_score(score_known, score_unknown)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        # roc_auc = auc(fpr, tpr)
        roc_auc = roc_auc_score(y_true, y_pred)

        plt.figure(figsize=(5, 5))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")
        plt.show()

    def get_rejection_threshold(self, pred_known, pred_unknown, tpr_percentage=0.9):
        # Get OSR scores from predictions
        score_known = self.osr_score(pred_known)
        score_unknown = self.osr_score(pred_unknown)
        y_true, y_pred = self._format_score(score_known, score_unknown)

        # Compute ROC thresholds from the scores
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # Search for the threshold that gives the desired TPR
        i = 0
        while tpr[i] < tpr_percentage:
            i += 1

        selected_threshold = thresholds[i] * np.sign(thresholds[i])

        print(
            f"For a TPR of {tpr[i]}, FPR is {fpr[i]} (i.e. {fpr[i]*100:.2f}% of knowns are classified as unknowns)"
        )
        print("Threshold:", selected_threshold)

        return selected_threshold
