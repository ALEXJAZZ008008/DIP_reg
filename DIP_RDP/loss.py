# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf
import tensorflow.keras as k


# https://github.com/keras-team/keras/blob/master/keras/losses.py#L1174-L1204
def mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    return k.backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)


# https://github.com/keras-team/keras/blob/master/keras/losses.py#L1580-L1617
def log_cosh(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    def _logcosh(x):
        return x + tf.math.softplus(-2. * x) - tf.cast(tf.math.log(2.), x.dtype)

    return k.backend.mean(_logcosh(y_pred - y_true), axis=-1)


# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
def correlation_coefficient_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    mx = k.backend.mean(y_true)
    my = k.backend.mean(y_pred)

    xm = y_true - mx
    ym = y_pred - my

    r_num = k.backend.sum(tf.multiply(xm, ym))
    r_den = k.backend.sqrt(tf.multiply(k.backend.sum(k.backend.square(xm)), k.backend.sum(k.backend.square(ym))))
    r = r_num / r_den

    r = k.backend.maximum(k.backend.minimum(r, 1.0), -1.0)

    return 1 - k.backend.square(r)


def accuracy_correlation_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    return (correlation_coefficient_loss(y_true, y_pred) * -1.0) + 1.0
