import tensorflow as tf


def ramp_loss(t, s, m):
    return tf.minimum(tf.cast(m, dtype=tf.float32) - s, tf.maximum(0.0, m - t))


def loss_compact_hyperspheres(class_anchors, nb_classes, m, s, _lambda, kappa):
    @tf.function
    def _loss(y_true, y_pred, write=False):
        # sometimes there are empty batches distributed to replicas
        batch_size = tf.shape(y_true)[0]
        if batch_size == 0:
            return tf.reshape(tf.constant(0, dtype=tf.float32), (-1,))

        c_yi = tf.gather(class_anchors, y_true)

        first_term = tf.norm(y_pred - c_yi, axis=1)

        distance_to_all_anchors = tf.norm(
            tf.expand_dims(y_pred, 1) - class_anchors, axis=2
        )
        # adding m first introduce approximation errors ???
        max_val = tf.maximum(
            0.0, tf.expand_dims(first_term, 1) - distance_to_all_anchors + m
        )
        # some indices are skipped in the sum operation
        indices_to_ignore = tf.stack(
            [tf.range(batch_size, dtype=tf.int64), y_true], axis=1
        )
        max_val = tf.tensor_scatter_nd_update(
            max_val, indices_to_ignore, tf.zeros(batch_size)
        )
        second_term = _lambda / nb_classes * tf.reduce_sum(max_val, axis=1)

        third_term = (
            kappa
            / tf.cast(batch_size, dtype=tf.float32)
            * tf.reduce_sum(
                ramp_loss(
                    distance_to_all_anchors - tf.expand_dims(first_term, 1), s, m
                ),
                axis=1,
            )
        )

        return first_term + second_term

    return _loss
