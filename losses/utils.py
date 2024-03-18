import tensorflow as tf


def softmin(x):
    return tf.math.exp(-x) / tf.math.reduce_sum(tf.math.exp(-x))


def distance(x, multiple_y, metric="euclidean"):
    # compute a distance between x and each element of multiple_y
    if metric == "euclidean":
        return tf.norm(tf.expand_dims(x, 1) - multiple_y, axis=2)
