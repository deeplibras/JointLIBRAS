import tensorflow as tf
from tflearn import utils

def euclidian(y_pred, y_true):
    with tf.name_scope("EuclidianDistance"):
        # return tf.reduce_sum(tf.square(y_pred - y_true))
        # return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_pred, y_true))))
        return tf.reduce_sum(tf.sqrt(tf.square(y_pred[:,::2] - y_true[:,::2])))
