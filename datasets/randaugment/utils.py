# Code from : https://github.com/tensorflow/models/blob/v2.13.0/official/vision/ops/augment.py
# Adapted to work on grayscale images like MNIST
#

import tensorflow as tf
import math
from typing import Any, List, Iterable, Optional, Tuple, Union

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

def to_4d(image: tf.Tensor) -> tf.Tensor:
    """Converts an input Tensor to 4 dimensions.

    4D image => [N, H, W, C] or [N, C, H, W]
    3D image => [1, H, W, C] or [1, C, H, W]
    2D image => [1, H, W, 1]

    Args:
        image: The 2/3/4D input tensor.

    Returns:
        A 4D image tensor.

    Raises:
        `TypeError` if `image` is not a 2/3/4D tensor.

    """
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)

def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
    """Converts a 4D image back to `ndims` rank."""
    shape = tf.shape(image)
    begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)

def _convert_translation_to_transform(translations: tf.Tensor) -> tf.Tensor:
  """Converts translations to a projective transform.

  The translation matrix looks like this:
    [[1 0 -dx]
     [0 1 -dy]
     [0 0 1]]

  Args:
    translations: The 2-element list representing [dx, dy], or a matrix of
      2-element lists representing [dx dy] to translate for each image. The
      shape must be static.

  Returns:
    The transformation matrix of shape (num_images, 8).

  Raises:
    `TypeError` if
      - the shape of `translations` is not known or
      - the shape of `translations` is not rank 1 or 2.

  """
  translations = tf.convert_to_tensor(translations, dtype=tf.float32)
  if translations.get_shape().ndims is None:
    raise TypeError('translations rank must be statically known')
  elif len(translations.get_shape()) == 1:
    translations = translations[None]
  elif len(translations.get_shape()) != 2:
    raise TypeError('translations should have rank 1 or 2.')
  num_translations = tf.shape(translations)[0]

  return tf.concat(
      values=[
          tf.ones((num_translations, 1), tf.dtypes.float32),
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          -translations[:, 0, None],
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          tf.ones((num_translations, 1), tf.dtypes.float32),
          -translations[:, 1, None],
          tf.zeros((num_translations, 2), tf.dtypes.float32),
      ],
      axis=1,
  )

def _convert_angles_to_transform(angles: tf.Tensor, image_width: tf.Tensor,
                                 image_height: tf.Tensor) -> tf.Tensor:
    """Converts an angle or angles to a projective transform.

    Args:
        angles: A scalar to rotate all images, or a vector to rotate a batch of
        images. This must be a scalar.
        image_width: The width of the image(s) to be transformed.
        image_height: The height of the image(s) to be transformed.

    Returns:
        A tensor of shape (num_images, 8).

    Raises:
        `TypeError` if `angles` is not rank 0 or 1.

    """
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    if len(angles.get_shape()) == 0:  # pylint:disable=g-explicit-length-test
        angles = angles[None]
    elif len(angles.get_shape()) != 1:
        raise TypeError('Angles should have a rank 0 or 1.')
    x_offset = ((image_width - 1) -
                (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) *
                (image_height - 1))) / 2.0
    y_offset = ((image_height - 1) -
                (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) *
                (image_height - 1))) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(
        values=[
            tf.math.cos(angles)[:, None],
            -tf.math.sin(angles)[:, None],
            x_offset[:, None],
            tf.math.sin(angles)[:, None],
            tf.math.cos(angles)[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.dtypes.float32),
        ],
        axis=1,
    )

def _apply_transform_to_images(
    images,
    transforms,
    fill_mode='reflect',
    fill_value=0.0,
    interpolation='bilinear',
    output_shape=None,
    name=None,
):
    """Applies the given transform(s) to the image(s).

    Args:
    images: A tensor of shape `(num_images, num_rows, num_columns,
      num_channels)` (NHWC). The rank must be statically known (the shape is
      not `TensorShape(None)`).
    transforms: Projective transform matrix/matrices. A vector of length 8 or
      tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1,
      b2, c0, c1], then it maps the *output* point `(x, y)` to a transformed
      *input* point `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) /
      k)`, where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared
      to the transform mapping input points to output points. Note that
      gradients are not backpropagated into transformation parameters.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
    fill_value: a float represents the value to be filled outside the
      boundaries when `fill_mode="constant"`.
    interpolation: Interpolation mode. Supported values: `"nearest"`,
      `"bilinear"`.
    output_shape: Output dimension after the transform, `[height, width]`. If
      `None`, output is the same size as input image.
    name: The name of the op.  Fill mode behavior for each valid value is as
      follows
      - `"reflect"`: `(d c b a | a b c d | d c b a)` The input is extended by
      reflecting about the edge of the last pixel.
      - `"constant"`: `(k k k k | a b c d | k k k k)` The input is extended by
      filling all values beyond the edge with the same constant value k = 0.
      - `"wrap"`: `(a b c d | a b c d | a b c d)` The input is extended by
      wrapping around to the opposite edge.
      - `"nearest"`: `(a a a a | a b c d | d d d d)` The input is extended by
      the nearest pixel.  Input shape: 4D tensor with shape:
      `(samples, height, width, channels)`, in `"channels_last"` format.
      Output shape: 4D tensor with shape: `(samples, height, width, channels)`,
      in `"channels_last"` format.

    Returns:
        Image(s) with the same type and shape as `images`, with the given
        transform(s) applied. Transformed coordinates outside of the input image
        will be filled with zeros.
    """
    with tf.name_scope(name or 'transform'):
        if output_shape is None:
            output_shape = tf.shape(images)[1:3]
        if not tf.executing_eagerly():
            output_shape_value = tf.get_static_value(output_shape)
            if output_shape_value is not None:
                output_shape = output_shape_value

        output_shape = tf.convert_to_tensor(
            output_shape, tf.int32, name='output_shape'
        )

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                'output_shape must be a 1-D Tensor of 2 elements: '
                'new_height, new_width, instead got '
                f'output_shape={output_shape}'
            )

        fill_value = tf.convert_to_tensor(fill_value, tf.float32, name='fill_value')

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )
  
def _fill_rectangle(image,
                    center_width,
                    center_height,
                    half_width,
                    half_height,
                    replace=None):
    """Fills blank area."""
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    lower_pad = tf.maximum(0, center_height - half_height)
    upper_pad = tf.maximum(0, image_height - center_height - half_height)
    left_pad = tf.maximum(0, center_width - half_width)
    right_pad = tf.maximum(0, image_width - center_width - half_width)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad)
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype),
        padding_dims,
        constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])

    if replace is None:
        fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
    elif isinstance(replace, tf.Tensor):
        fill = replace
    else:
        fill = tf.ones_like(image, dtype=image.dtype) * replace
    image = tf.where(tf.equal(mask, 0), fill, image)

    return image

def transform(
    image: tf.Tensor,
    transforms: Any,
    interpolation: str = 'nearest',
    output_shape=None,
    fill_mode: str = 'reflect',
    fill_value: float = 0.0,
) -> tf.Tensor:
    """Transforms an image."""
    original_ndims = tf.rank(image)
    transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    if transforms.shape.rank == 1:
        transforms = transforms[None]
    image = to_4d(image)
    image = _apply_transform_to_images(
        images=image,
        transforms=transforms,
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
        output_shape=output_shape,
    )
    return from_4d(image, original_ndims)


def translate(
    image: tf.Tensor,
    translations,
    fill_value: float = 0.0,
    fill_mode: str = 'reflect',
    interpolation: str = 'nearest',
) -> tf.Tensor:
    """Translates image(s) by provided vectors.

    Args:
        image: An image Tensor of type uint8.
        translations: A vector or matrix representing [dx dy].
        fill_value: a float represents the value to be filled outside the boundaries
        when `fill_mode="constant"`.
        fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
        interpolation: Interpolation mode. Supported values: `"nearest"`,
        `"bilinear"`.

    Returns:
        The translated version of the image.
    """
    transforms = _convert_translation_to_transform(translations)  # pytype: disable=wrong-arg-types  # always-use-return-annotations
    return transform(
        image,
        transforms=transforms,
        interpolation=interpolation,
        fill_value=fill_value,
        fill_mode=fill_mode,
    )

def rotate(image: tf.Tensor, degrees: float) -> tf.Tensor:
    """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
        image: An image Tensor of type uint8.
        degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.

    Returns:
        The rotated version of image.

    """
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = tf.cast(degrees * degrees_to_radians, tf.float32)

    original_ndims = tf.rank(image)
    image = to_4d(image)

    image_height = tf.cast(tf.shape(image)[1], tf.float32)
    image_width = tf.cast(tf.shape(image)[2], tf.float32)
    transforms = _convert_angles_to_transform(
        angles=radians, image_width=image_width, image_height=image_height)
    # In practice, we should randomize the rotation degrees by flipping
    # it negatively half the time, but that's done on 'degrees' outside
    # of the function.
    image = transform(image, transforms=transforms)
    return from_4d(image, original_ndims)

def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
        image1: An image Tensor of type uint8.
        image2: An image Tensor of type uint8.
        factor: A floating point value above 0.0.

    Returns:
        A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)

def _translate_level_to_arg(level: float, translate_const: float):
  level = (level / _MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)

def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level: float):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level: float):
  return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)

def _mult_to_arg(level: float, multiplier: float = 1.):
  return (int((level / _MAX_LEVEL) * multiplier),)

def level_to_arg(cutout_const: float, translate_const: float):
  """Creates a dict mapping image operation names to their arguments."""

  no_arg = lambda level: ()
  posterize_arg = lambda level: _mult_to_arg(level, 4)
  solarize_arg = lambda level: _mult_to_arg(level, 256)
  solarize_add_arg = lambda level: _mult_to_arg(level, 110)
  cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
  translate_arg = lambda level: _translate_level_to_arg(level, translate_const)
  translate_bbox_arg = lambda level: _translate_level_to_arg(level, 120)

  args = {
      'AutoContrast': no_arg,
      'Equalize': no_arg,
      'Invert': no_arg,
      'Rotate': _rotate_level_to_arg,
      'Posterize': posterize_arg,
      'Solarize': solarize_arg,
      'SolarizeAdd': solarize_add_arg,
      'Color': _enhance_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      'Cutout': cutout_arg,
      'TranslateX': translate_arg,
      'TranslateY': translate_arg,
      'Grayscale': no_arg,
  }
  return args

def bbox_wrapper(func):
  """Adds a bboxes function argument to func and returns unchanged bboxes."""
  def wrapper(images, bboxes, *args, **kwargs):
    return (func(images, *args, **kwargs), bboxes)
  return wrapper