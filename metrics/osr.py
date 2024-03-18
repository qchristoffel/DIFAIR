# code from https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/methods/ARPL/core/evaluation.py

import os
import sys

import numpy as np


def get_curve_online(known, novel, stypes=["Bas"]):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1 :] = tp[stype][l]
                fp[stype][l + 1 :] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1 :] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1 :] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - 0.95).argmin()
        tnr_at_tpr95[stype] = 1.0 - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(x1, x2, stypes=["Bas"], verbose=True):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ["TNR", "AUROC", "DTACC", "AUIN", "AUOUT"]
    if verbose:
        print("\t      ", end="")
        for mtype in mtypes:
            print(" {mtype:6s}".format(mtype=mtype), end="")
        print("")

    for stype in stypes:
        if verbose:
            print("\t{stype:5s} ".format(stype=stype), end="")
        results[stype] = dict()

        # TNR
        mtype = "TNR"
        results[stype][mtype] = 100.0 * tnr_at_tpr95[stype]
        if verbose:
            print(" {val:6.3f}".format(val=results[stype][mtype]), end="")

        # AUROC
        mtype = "AUROC"
        tpr = np.concatenate([[1.0], tp[stype] / tp[stype][0], [0.0]])
        fpr = np.concatenate([[1.0], fp[stype] / fp[stype][0], [0.0]])
        results[stype][mtype] = 100.0 * (-np.trapz(1.0 - fpr, tpr))
        if verbose:
            print(" {val:6.3f}".format(val=results[stype][mtype]), end="")

        # DTACC
        mtype = "DTACC"
        results[stype][mtype] = 100.0 * (
            0.5 * (tp[stype] / tp[stype][0] + 1.0 - fp[stype] / fp[stype][0]).max()
        )
        if verbose:
            print(" {val:6.3f}".format(val=results[stype][mtype]), end="")

        # AUIN
        mtype = "AUIN"
        denom = tp[stype] + fp[stype]
        denom[denom == 0.0] = -1.0
        pin_ind = np.concatenate([[True], denom > 0.0, [True]])
        pin = np.concatenate([[0.5], tp[stype] / denom, [0.0]])
        results[stype][mtype] = 100.0 * (-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(" {val:6.3f}".format(val=results[stype][mtype]), end="")

        # AUOUT
        mtype = "AUOUT"
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.0] = -1.0
        pout_ind = np.concatenate([[True], denom > 0.0, [True]])
        pout = np.concatenate([[0.0], (fp[stype][0] - fp[stype]) / denom, [0.5]])
        results[stype][mtype] = 100.0 * (np.trapz(pout[pout_ind], 1.0 - fpr[pout_ind]))
        if verbose:
            print(" {val:6.3f}".format(val=results[stype][mtype]), end="")
            print("")

    return results


def compute_oscr(pred_k, pred_u, labels, loss_helper):
    x1, x2 = loss_helper.osr_score(pred_k), loss_helper.osr_score(pred_u)
    x1 = x1.numpy()
    x2 = x2.numpy()

    pred = loss_helper.predicted_class(pred_k)

    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0.0 for x in range(n + 2)]
    FPR = [0.0 for x in range(n + 2)]

    reverse = -1 if loss_helper.score_type == "min" else 1
    idx = predict.argsort()[::reverse]
    # de base trié en décroissant, quand le score est

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1 :].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)
    # print(ROC)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR


def evaluation(pred_known, pred_unknown, labels, loss_helper):
    # print("Sklearn AUROC: {:.3f}".format(loss_helper.auroc(pred_known, pred_unknown)))

    # Out-of-Distribution detection evaluation
    print("\tEvaluation using maximum value of prediction")
    x1, x2 = np.max(pred_known, axis=1), np.max(pred_unknown, axis=1)
    results = metric_ood(x1, x2)["Bas"]
    # results['AUROC'] gives the auroc using the maximum value of the prediction
    auroc_max_val = results["AUROC"]

    print("\tEvaluation using defined osr score")
    x1, x2 = (
        loss_helper.osr_score(pred_known).numpy(),
        loss_helper.osr_score(pred_unknown).numpy(),
    )
    results = metric_ood(x1, x2)["Bas"]
    results["max_val_auroc"] = auroc_max_val

    # OSCR
    _oscr_score = compute_oscr(pred_known, pred_unknown, labels, loss_helper)

    results["acc"] = np.mean(loss_helper.predicted_class(pred_known) == labels)
    results["oscr"] = _oscr_score * 100.0
    results["real_auroc"] = loss_helper.auroc(pred_known, pred_unknown) * 100.0
    # results['another_auroc'] = loss_helper.auroc_v2(pred_known, pred_unknown) * 100.

    template = (
        "\tAcc: {:.3f} OSCR : {:.3f} AUROC_max_val (%): {:.3f} AUROC_real (%): {:.3f}"
    )
    print(
        template.format(
            results["acc"],
            results["oscr"],
            results["max_val_auroc"],
            results["real_auroc"],
        )
    )
    # print("Another AUROC: {:.3f}".format(results['another_auroc']))

    return results
