def modify_score(score_known, score_unknown, score_type):
    """Modify the scores for AUROC calculation, we want to have score of known values
    as low as possible and score of unknown values as high as possible. This is because
    for AUROC we use 0 as the target for known values and 1 as the target for unknown values.
    """
    if score_type == "min":
        # the lower the score, the more likely it is to be known
        return score_known, score_unknown
    elif score_type == "max":
        # the higher the score, the more likely it is to be known
        # so we invert the sign
        return -score_known, -score_unknown
