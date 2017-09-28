import tensorflow as tf
from tflearn import utils

def euclidian(y_pred, y_true):
    with tf.name_scope("EuclidianDistance"):
        return tf.reduce_mean(tf.reduce_mean([
            tf.add(
                tf.square(tf.subtract(y_pred[:, x], y_true[:, x])),
                tf.square(tf.subtract(y_pred[:, x+1], y_true[:, x+1]))
            )
            for x in range(0, y_true.shape[-1], 2)
        ], 1))
