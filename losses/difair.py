import tensorflow as tf


def hard_loss(class_anchors, max_dist):

    anchors_dist = tf.norm(class_anchors[0] - class_anchors[1])

    @tf.function
    def _loss(y_true, y_pred):
        # sometimes there are empty batches distributed to replicas
        batch_size = tf.shape(y_true)[0]
        if batch_size == 0:
            return tf.reshape(tf.constant(0, dtype=tf.float32), (-1,))

        # y true is not one hot encoded
        representation_to_learn = tf.gather(class_anchors, y_true)

        ## Distance to anchor
        dist = tf.norm(y_pred - representation_to_learn, axis=1)
        distance_to_all_anchors = tf.norm(
            tf.expand_dims(y_pred, 1) - class_anchors, axis=2
        )

        relax_dist = tf.maximum(dist - max_dist, 0)
        tf.summary.scalar("loss/dist", tf.reduce_mean(relax_dist))

        dist_difference = tf.expand_dims(dist, axis=1) - distance_to_all_anchors
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
        # tf.summary.scalar("loss/tuplet_loss", tf.reduce_mean(tuplet_loss))

        return relax_dist
        # return 0.1 * relax_dist + tuplet_loss

    return _loss


def smooth_loss(class_anchors, max_dist):

    anchors_dist = tf.norm(class_anchors[0] - class_anchors[1])

    @tf.function
    def _loss(y_true, y_pred):
        # sometimes there are empty batches distributed to replicas
        batch_size = tf.shape(y_true)[0]
        if batch_size == 0:
            return tf.reshape(tf.constant(0, dtype=tf.float32), (-1,))

        # y true is not one hot encoded
        representation_to_learn = tf.gather(class_anchors, y_true)

        ## Distance to anchor
        dist = tf.norm(y_pred - representation_to_learn, axis=1)
        distance_to_all_anchors = tf.norm(
            tf.expand_dims(y_pred, 1) - class_anchors, axis=2
        )

        relax_dist = 0.1 * dist + tf.maximum(dist - max_dist, 0)
        tf.summary.scalar("loss/dist", tf.reduce_mean(relax_dist))

        dist_difference = tf.expand_dims(dist, axis=1) - distance_to_all_anchors
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
        # tf.summary.scalar("loss/tuplet_loss", tf.reduce_mean(tuplet_loss))

        return relax_dist
        # return 0.1 * relax_dist + tuplet_loss

    return _loss
