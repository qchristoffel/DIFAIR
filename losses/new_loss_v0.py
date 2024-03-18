import tensorflow as tf


def loss_belong(nb_classes, nb_features, anchor_multiplier):
    @tf.function
    def _loss(y_true, x, write=False):
        batch_size = tf.shape(y_true)[0]
        if batch_size == 0:
            return tf.reshape(tf.constant(0, dtype=tf.float32), (-1,))

        # tf.print(x)
        # return tf.zeros(tf.shape(y_true)[0], dtype=tf.float32)
        x = tf.reshape(x, (-1, nb_classes, nb_features))
        x_count = tf.reduce_sum(tf.where(x > 0.5, 1.0, 0.0), axis=2)
        x_magnitude = tf.norm(x, axis=2)

        # on veut que x_count[y_true] soit maximal et si les autres valeurs sont
        # activées c'est pas grave jusqu'à un certain seuil
        activated_count_threshold = (
            1  # voir si besoin d'ajouter ça ou pas (c'est si on veut accepter
        )
        # qu'on ait des valeurs activées sur les autres classes, en pénalisant pas du tout si on a ce
        # nombre d'activations)

        # ici on accepte plus ou moins le nombre de features activées par classe
        decrease = lambda x: nb_features**2 * ((x / nb_features) - 1) ** 4
        increase = lambda x: nb_features * ((x / nb_features)) ** 2
        # tmp = decrease(x_count[y_true])
        tmp = decrease(
            tf.gather_nd(
                x_count,
                tf.stack(
                    [tf.range(tf.shape(y_true)[0], dtype=tf.int64), y_true], axis=1
                ),
            )
        )
        x_count = increase(x_count)
        # x_count[y_true] = tmp
        x_count = tf.tensor_scatter_nd_update(
            x_count,
            tf.stack([tf.range(tf.shape(y_true)[0], dtype=tf.int64), y_true], axis=1),
            tmp,
        )

        # revoir, on veut que la magnitude de la vraie classe soit supérieure aux autres
        max_magnitude = tf.norm(tf.constant([1.0] * nb_features) * anchor_multiplier)
        decrease = lambda x: max_magnitude * ((x / max_magnitude) - 1) ** 4
        increase = lambda x: max_magnitude * ((x / max_magnitude)) ** 2
        # tmp = decrease(x_magnitude[y_true])
        tmp = decrease(
            tf.gather_nd(
                x_magnitude,
                tf.stack(
                    [tf.range(tf.shape(y_true)[0], dtype=tf.int64), y_true], axis=1
                ),
            )
        )
        x_magnitude = increase(x_magnitude)
        # x_magnitude[y_true] = tmp
        x_magnitude = tf.tensor_scatter_nd_update(
            x_magnitude,
            tf.stack([tf.range(tf.shape(y_true)[0], dtype=tf.int64), y_true], axis=1),
            tmp,
        )

        # print(x_count)
        # print(x_magnitude)

        return tf.reduce_sum(x_count, axis=1) + tf.reduce_sum(x_magnitude, axis=1)

    return _loss
