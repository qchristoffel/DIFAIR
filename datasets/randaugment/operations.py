# Code from : https://github.com/tensorflow/models/blob/v2.13.0/official/vision/ops/augment.py
# Adapted to work on grayscale images like MNIST
#

import tensorflow as tf
import math
from typing import Any, List, Iterable, Optional, Tuple, Union

from datasets.randaugment.utils import blend, translate, transform, rotate, _fill_rectangle

def autocontrast(image: tf.Tensor) -> tf.Tensor:
    """Implements Autocontrast function from PIL using TF ops.

    Args:
        image: A 3D uint8 tensor.

    Returns:
        The image after it has had autocontrast applied to it and will be of type
        uint8.
    """

    def scale_channel(image: tf.Tensor) -> tf.Tensor:
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[..., 0])
    s2 = scale_channel(image[..., 1])
    s3 = scale_channel(image[..., 2])
    image = tf.stack([s1, s2, s3], -1)
    return image
    
    # def scale_all(image: tf.Tensor) -> tf.Tensor:
    #     s1 = scale_channel(image[..., 0])
    #     s2 = scale_channel(image[..., 1])
    #     s3 = scale_channel(image[..., 2])
    #     image = tf.stack([s1, s2, s3], -1)
    #     return image
    
    # image = tf.cond(tf.shape(image)[-1] > 1, 
    #                 lambda: scale_channel(image[..., 0]), 
    #                 lambda: scale_all(image))

    # return image


def sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image
    image = tf.cast(image, tf.float32)
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    if orig_image.shape.rank == 3:
        kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                            dtype=tf.float32,
                            shape=[3, 3, 1, 1]) / 13.
        # Tile across channel dimension.
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        strides = [1, 1, 1, 1]
        degenerate = tf.nn.depthwise_conv2d(
            image, kernel, strides, padding='VALID', dilations=[1, 1])
    elif orig_image.shape.rank == 4:
        kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                            dtype=tf.float32,
                            shape=[1, 3, 3, 1, 1]) / 13.
        strides = [1, 1, 1, 1, 1]
        # Run the kernel across each channel
        channels = tf.split(image, 3, axis=-1)
        degenerates = [
            tf.nn.conv3d(channel, kernel, strides, padding='VALID',
                        dilations=[1, 1, 1, 1, 1])
            for channel in channels
        ]
        degenerate = tf.concat(degenerates, -1)
    else:
        raise ValueError('Bad image rank: {}'.format(image.shape.rank))
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf.ones_like(degenerate)
    paddings = [[0, 0]] * (orig_image.shape.rank - 3)
    padded_mask = tf.pad(mask, paddings + [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, paddings + [[1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return blend(result, orig_image, factor)


def equalize(image: tf.Tensor) -> tf.Tensor:
    """Implements Equalize function from PIL using TF ops."""

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[..., c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(
            tf.equal(step, 0), lambda: im,
            lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], -1)
    return image


def invert(image: tf.Tensor) -> tf.Tensor:
    """Inverts the image pixels."""
    image = tf.convert_to_tensor(image)
    return 255 - image

def wrap(image: tf.Tensor) -> tf.Tensor:
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf.shape(image)
    extended_channel = tf.expand_dims(tf.ones(shape[:-1], image.dtype), -1)
    extended = tf.concat([image, extended_channel], axis=-1)
    return extended


def unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
    """Unwraps an image produced by wrap.

    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set to 128).  Operations like translate and shear on a wrapped
    Tensor will leave 0s in empty locations.  Some transformations look
    at the intensity of values to do preprocessing, and we want these
    empty pixels to assume the 'average' value, rather than pure black.


    Args:
        image: A 3D Image Tensor with 4 channels.
        replace: A one or three value 1D tensor to fill empty pixels.

    Returns:
        image: A 3D image Tensor with 3 channels.
    """
    image_shape = tf.shape(image)
    # Flatten the spatial dimensions.
    flattened_image = tf.reshape(image, [-1, image_shape[-1]])

    # Find all pixels where the last channel is zero.
    alpha_channel = tf.expand_dims(flattened_image[..., 3], axis=-1)

    replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    flattened_image = tf.where(
        tf.equal(alpha_channel, 0),
        tf.ones_like(flattened_image, dtype=image.dtype) * replace,
        flattened_image)

    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(
        image,
        [0] * image.shape.rank,
        tf.concat([image_shape[:-1], [3]], -1))
    return image

def wrapped_rotate(image: tf.Tensor, degrees: float, replace: int) -> tf.Tensor:
    """Applies rotation with wrap/unwrap."""
    image = rotate(wrap(image), degrees=degrees)
    return unwrap(image, replace)


def translate_x(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
    """Equivalent of PIL Translate in X dimension."""
    image = translate(wrap(image), [-pixels, 0])
    return unwrap(image, replace)


def translate_y(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
    """Equivalent of PIL Translate in Y dimension."""
    image = translate(wrap(image), [0, -pixels])
    return unwrap(image, replace)


def shear_x(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    image = transform(
        image=wrap(image), transforms=[1., level, 0., 0., 1., 0., 0., 0.])
    return unwrap(image, replace)


def shear_y(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    image = transform(
        image=wrap(image), transforms=[1., 0., 0., level, 1., 0., 0., 0.])
    return unwrap(image, replace)

def contrast(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Contrast."""
    degenerate = tf.image.rgb_to_grayscale(image)
    # Cast before calling tf.histogram.
    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
    return blend(degenerate, image, factor)


def brightness(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)


def posterize(image: tf.Tensor, bits: int) -> tf.Tensor:
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)

def solarize(image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
    """Solarize the input image(s)."""
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return tf.where(image < threshold, image, 255 - image)


def solarize_add(image: tf.Tensor,
                 addition: int = 0,
                 threshold: int = 128) -> tf.Tensor:
    """Additive solarize the input image(s)."""
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    return tf.where(image < threshold, added_image, image)

def grayscale(image: tf.Tensor) -> tf.Tensor:
  """Convert image to grayscale."""
  
  return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))

def color(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Color."""
    degenerate = grayscale(image)
    return blend(degenerate, image, factor)

def cutout(image: tf.Tensor, pad_size: int, replace: int = 0) -> tf.Tensor:
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `image`. The pixel values filled in will be of the
    value `replace`. The location where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
        image: An image Tensor of type uint8.
        pad_size: Specifies how big the zero mask that will be generated is that is
        applied to the image. The mask will be of size (2*pad_size x 2*pad_size).
        replace: What pixel value to fill in the image in the area that has the
        cutout mask applied to it.

    Returns:
        An image Tensor that is of type uint8.
    """
    if image.shape.rank not in [3, 4]:
        raise ValueError('Bad image rank: {}'.format(image.shape.rank))

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(
        shape=[], minval=0, maxval=image_height, dtype=tf.int32)

    cutout_center_width = tf.random.uniform(
        shape=[], minval=0, maxval=image_width, dtype=tf.int32)

    image = _fill_rectangle(image, cutout_center_width, cutout_center_height,
                            pad_size, pad_size, replace)

    return image

NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': wrapped_rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'Cutout': cutout,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout'
})