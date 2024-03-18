# Code from : https://github.com/tensorflow/models/blob/v2.13.0/official/vision/ops/augment.py
# Adapted to work on grayscale images like MNIST
#

import inspect
import tensorflow as tf
import math
from typing import Any, List, Iterable, Optional, Tuple, Union

from datasets.randaugment.operations import NAME_TO_FUNC, REPLACE_FUNCS
from datasets.randaugment.utils import _MAX_LEVEL, level_to_arg, bbox_wrapper

def _parse_policy_info(name: str,
                       prob: float,
                       level: float,
                       replace_value: List[int],
                       cutout_const: float,
                       translate_const: float,
                       level_std: float = 0.) -> Tuple[Any, float, Any]:
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]

    if level_std > 0:
        level += tf.random.normal([], dtype=tf.float32)
        level = tf.clip_by_value(level, 0., _MAX_LEVEL)

    args = level_to_arg(cutout_const, translate_const)[name](level)

    if name in REPLACE_FUNCS:
        # Add in replace arg if it is required for the function that is called.
        args = tuple(list(args) + [replace_value])

    # Add bboxes as the second positional argument for the function if it does
    # not already exist.
    if 'bboxes' not in inspect.getfullargspec(func)[0]:
        func = bbox_wrapper(func)

    return func, prob, args

def _maybe_identity(x: Optional[tf.Tensor]) -> Optional[tf.Tensor]:
    return tf.identity(x) if x is not None else None

class RandAugment():
    """Applies the RandAugment policy to images.

    RandAugment is from the paper https://arxiv.org/abs/1909.13719.
    """

    def __init__(self,
                num_layers: int = 2,
                magnitude: float = 10.,
                cutout_const: float = 40.,
                translate_const: float = 100.,
                magnitude_std: float = 0.0,
                prob_to_apply: Optional[float] = None,
                exclude_ops: Optional[List[str]] = None):
        """Applies the RandAugment policy to images.

        Args:
        num_layers: Integer, the number of augmentation transformations to apply
            sequentially to an image. Represented as (N) in the paper. Usually best
            values will be in the range [1, 3].
        magnitude: Integer, shared magnitude across all augmentation operations.
            Represented as (M) in the paper. Usually best values are in the range
            [5, 10].
        cutout_const: multiplier for applying cutout.
        translate_const: multiplier for applying translation.
        magnitude_std: randomness of the severity as proposed by the authors of
            the timm library.
        prob_to_apply: The probability to apply the selected augmentation at each
            layer.
        exclude_ops: exclude selected operations.
        """
        super(RandAugment, self).__init__()

        self.num_layers = num_layers
        self.magnitude = float(magnitude)
        self.cutout_const = float(cutout_const)
        self.translate_const = float(translate_const)
        self.prob_to_apply = (
            float(prob_to_apply) if prob_to_apply is not None else None)
        self.available_ops = [
            'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
            'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
            'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd'
        ]
        self.magnitude_std = magnitude_std
        if exclude_ops:
            self.available_ops = [
                op for op in self.available_ops if op not in exclude_ops
            ]

    def _distort_common(
        self,
        image: tf.Tensor,
        bboxes: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Distorts the image and optionally bounding boxes."""
        input_image_type = image.dtype

        if input_image_type != tf.uint8:
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = tf.cast(image, dtype=tf.uint8)

        replace_value = [128] * 3
        min_prob, max_prob = 0.2, 0.8

        aug_image = image
        aug_bboxes = bboxes

        for _ in range(self.num_layers):
            op_to_select = tf.random.uniform([],
                                             maxval=len(self.available_ops) + 1,
                                             dtype=tf.int32)

            branch_fns = []
            for (i, op_name) in enumerate(self.available_ops):
                prob = tf.random.uniform([],
                                        minval=min_prob,
                                        maxval=max_prob,
                                        dtype=tf.float32)
                func, _, args = _parse_policy_info(op_name, prob, self.magnitude,
                                                replace_value, self.cutout_const,
                                                self.translate_const,
                                                self.magnitude_std)
                branch_fns.append((
                    i,
                    # pylint:disable=g-long-lambda
                    lambda selected_func=func, selected_args=args: selected_func(
                        image, bboxes, *selected_args)))
                # pylint:enable=g-long-lambda

            aug_image, aug_bboxes = tf.switch_case(
                branch_index=op_to_select,
                branch_fns=branch_fns,
                default=lambda: (tf.identity(image), _maybe_identity(bboxes)))  # pylint: disable=cell-var-from-loop

            if self.prob_to_apply is not None:
                aug_image, aug_bboxes = tf.cond(
                    tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
                    lambda: (tf.identity(aug_image), _maybe_identity(aug_bboxes)),
                    lambda: (tf.identity(image), _maybe_identity(bboxes)))
            image = aug_image
            bboxes = aug_bboxes

        image = tf.cast(image, dtype=input_image_type)
        return image, bboxes

    def distort(self, image: tf.Tensor) -> tf.Tensor:
        """See base class."""
        image, _ = self._distort_common(image)
        return image
    
class RandAugmentLayer(tf.keras.layers.Layer):
    
    def __init__(self, n=1, m=6, **kwargs):
        super(RandAugmentLayer, self).__init__()
        self.n = n
        self.m = m
        self.augment = RandAugment(num_layers=n, magnitude=m, **kwargs)
        
    def call(self, inputs):    
        return tf.map_fn(lambda images: self.augment.distort(images), inputs, parallel_iterations=32)
