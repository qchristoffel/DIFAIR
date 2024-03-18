import numpy as np

from helpers.cac import CACHelper
from helpers.compact_hypersphere import CompactHypersphereHelper
from helpers.crossentropy import CrossEntropyHelper
from helpers.difair import DifairHelper
from helpers.new_loss_v0 import NewPropositionHelper0
from helpers.new_loss_v1 import NewPropositionHelper1
from helpers.new_loss_v2 import NewPropositionHelper2
from helpers.new_loss_v3 import NewPropositionHelper3


def get_loss_helper(args, class_anchors, nb_classes):
    max_angle = np.cos(args.max_angle * np.pi / 180)

    print("Training with loss:", args.loss)

    if args.loss == "crossentropy":
        return CrossEntropyHelper(args.osr_score, args.use_softmax)
    elif args.loss == "cac":
        return CACHelper(class_anchors, nb_classes, args.nb_features)
    elif args.loss == "difair":
        return DifairHelper(
            class_anchors,
            args.max_dist,
            nb_classes,
            args.nb_features,
            difair_type=args.difair_loss_type,
            osr_score=args.osr_score,
            reconstruction_weight=(
                args.reconstruction_weight if args.reconstruction else 0
            ),
        )
    # elif args.loss == "penalize_wrong":
    #     return penalize_wrong_classification(class_anchors, args.max_dist)
    # elif args.loss == "individual_dimensions":
    #     return loss_individual_dimensions(class_anchors,
    #                                       nb_classes,
    #                                       args.nb_features,
    #                                       args.max_dist)
    elif args.loss == "compact_hypersphere":
        distance_between_anchors = np.linalg.norm(class_anchors[0] - class_anchors[1])
        return CompactHypersphereHelper(
            class_anchors,
            nb_classes,
            args.osr_score,
            m=distance_between_anchors / 2,
            s=1,
            _lambda=0.1,
            kappa=0.5,
        )
    elif args.loss == "new_v0":
        return NewPropositionHelper0(
            nb_classes, args.nb_features, args.anchor_multiplier
        )
    elif args.loss == "new_v1":
        return NewPropositionHelper1(
            nb_classes, args.nb_features, args.anchor_multiplier
        )
    elif args.loss == "new_v2":
        return NewPropositionHelper2(
            nb_classes, args.nb_features, args.anchor_multiplier
        )
    elif args.loss == "new_v3":
        return NewPropositionHelper3(
            nb_classes, args.nb_features, args.anchor_multiplier
        )
    else:
        raise ValueError(f"Loss {args.loss} not implemented")
