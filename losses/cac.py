import tensorflow as tf


def loss_cac(class_anchors):
    @tf.function
    def _loss(y_true, y_pred, write=False):
        # sometimes there are empty batches distributed to replicas
        batch_size = tf.shape(y_true)[0]
        if batch_size == 0:
            return tf.reshape(tf.constant(0, dtype=tf.float32), (-1,))

        representation_to_learn = tf.gather(class_anchors, y_true)

        anchor_loss = tf.norm(y_pred - representation_to_learn, axis=1)

        distance_to_all_anchors = tf.norm(
            tf.expand_dims(y_pred, 1) - class_anchors, axis=2
        )

        dist_difference = tf.expand_dims(anchor_loss, axis=1) - distance_to_all_anchors
        exp_difference = tf.exp(dist_difference)
        # some indices should be skipped in the sum operation so set them to 0 before sum
        indices_to_ignore = tf.stack(
            [tf.range(batch_size, dtype=tf.int64), y_true], axis=1
        )
        dist_difference = tf.tensor_scatter_nd_update(
            exp_difference, indices_to_ignore, tf.zeros(batch_size)
        )
        difference_sums = tf.reduce_sum(exp_difference, axis=1)

        tuplet_loss = tf.math.log(1 + difference_sums)

        return 0.1 * anchor_loss + tuplet_loss

    return _loss
