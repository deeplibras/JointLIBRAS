import tensorflow as tf
from tflearn import utils
# import tensorflow.contrib.layers.flatten as flatten

def euclidean(y_pred, y_true):
    with tf.name_scope("EuclidianDistance"):
        return tf.reduce_mean(tf.reduce_mean([
            tf.add(
                tf.square(tf.subtract(y_pred[:, x], y_true[:, x])),
                tf.square(tf.subtract(y_pred[:, x+1], y_true[:, x+1]))
            )
            for x in range(0, y_true.shape[-1], 2)
        ], 1))

def corrected_euclidean(y_pred, y_true):
    with tf.name_scope("EuclideanDistance"):
        return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.reduce_sum([
            # the [-1] reshape flattens the tensor (it`s in the docs)
            tf.square(tf.subtract(tf.reshape(y_pred,[-1]), tf.reshape(y_true,[-1])))
        ], 1))))

def manhatan(y_pred, y_true):
    with tf.name_scope("ManhatanDistance"):
        return tf.reduce_mean(tf.reduce_mean([
            tf.add(
                tf.abs(tf.subtract(y_pred[:, x], y_true[:, x])),
                tf.abs(tf.subtract(y_pred[:, x+1], y_true[:, x+1]))
            )
            for x in range(0, y_true.shape[-1], 2)
        ], 1))

# def euclidean2():
#     with tf.name_scope("EuclideanDistance"):
#     euclidean_loss = tf.sqrt(tf.reduce_sum(
#               tf.square(tf.subtract(flatten_predicts, flatten_labels)), 1))
#             euclidean_loss_mean = tf.reduce_mean(euclidean_loss)
