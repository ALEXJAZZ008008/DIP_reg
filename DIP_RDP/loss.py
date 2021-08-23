# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import math
import tensorflow as tf
import tensorflow.keras as k


# https://github.com/keras-team/keras/blob/master/keras/losses.py#L1580-L1617
def log_cosh_loss(y_true, y_pred):
    def _log_cosh(x):
        return (x + tf.math.softplus(-2.0 * x)) - tf.cast(tf.math.log(2.0), x.dtype)

    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    return k.backend.mean(_log_cosh(y_pred - y_true), axis=-1)


def total_variation_loss(_, y_pred):
    def _pixel_dif_negative(x, n1, n2, n3):
        return tf.math.reduce_mean((1.0 / math.sqrt(math.pow(n1, 2.0) * math.pow(n2, 2.0) * math.pow(n3, 2.0))) *
                                   tf.math.pow(x[:, :-n1 or None, :-n2 or None, :-n3 or None, :] -
                                               x[:, n1 or None:, n2 or None:, n3 or None:, :], 2.0))

    def _pixel_dif_positive(x, n1, n2, n3):
        return tf.math.reduce_mean((1.0 / math.sqrt(math.pow(n1, 2.0) * math.pow(n2, 2.0) * math.pow(n3, 2.0))) *
                                   tf.math.pow(x[:, n1 or None:, n2 or None:, n3 or None:, :] -
                                               x[:, :-n1 or None, :-n2 or None, :-n3 or None, :], 2.0))

    def _total_variation_loss(x, n, _pixel_dif):
        _pixel_dif.append(_pixel_dif_negative(x, n, 0, 0))
        _pixel_dif.append(_pixel_dif_negative(x, 0, n, 0))
        _pixel_dif.append(_pixel_dif_negative(x, 0, 0, n))

        _pixel_dif.append(_pixel_dif_negative(x, n, n, 0))
        _pixel_dif.append(_pixel_dif_negative(x, n, 0, n))
        _pixel_dif.append(_pixel_dif_negative(x, 0, n, n))

        _pixel_dif.append(_pixel_dif_negative(x, n, n, n))

        _pixel_dif.append(_pixel_dif_positive(x, n, 0, 0))
        _pixel_dif.append(_pixel_dif_positive(x, 0, n, 0))
        _pixel_dif.append(_pixel_dif_positive(x, 0, 0, n))

        _pixel_dif.append(_pixel_dif_positive(x, n, n, 0))
        _pixel_dif.append(_pixel_dif_positive(x, n, 0, n))
        _pixel_dif.append(_pixel_dif_positive(x, 0, n, n))

        _pixel_dif.append(_pixel_dif_positive(x, n, n, n))

        return _pixel_dif

    y_pred = tf.cast(y_pred, dtype=tf.float64)

    # The input is a batch of images with shape:
    # [batch, height, width, depth, channels].

    y_pred = tf.pad(y_pred, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")

    pixel_dif = []

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height, width and depth by slicing.
    # Calculate the total variation by taking the absolute value of the
    # pixel-differences summing over the appropriate axis.
    pixel_dif = _total_variation_loss(y_pred, 1, pixel_dif)

    return tf.reduce_mean(tf.convert_to_tensor(pixel_dif))


def log_cosh_total_variation_loss(y_true, y_pred):
    return ((1.0 * log_cosh_loss(y_true, y_pred)) + (1e-01 * total_variation_loss(y_true, y_pred))) / 2.0


def relative_difference_loss(_, y_pred):
    def _pixel_dif_negative(x, n1, n2, n3, _gamma):
        current_pixel_dif = x[:, :-n1 or None, :-n2 or None, :-n3 or None, :] - \
                            x[:, n1 or None:, n2 or None:, n3 or None:, :]

        return tf.reduce_mean((1.0 / math.sqrt(math.pow(n1, 2.0) * math.pow(n2, 2.0) * math.pow(n3, 2.0))) *
                              (tf.math.pow(current_pixel_dif, 2.0) /
                               ((x[:, :-n1 or None, :-n2 or None, :-n3 or None, :] +
                                 x[:, n1 or None:, n2 or None:, n3 or None:, :]) +
                                (_gamma * tf.math.abs(current_pixel_dif)))))

    def _pixel_dif_positive(x, n1, n2, n3, _gamma):
        current_pixel_dif = x[:, n1 or None:, n2 or None:, n3 or None:, :] - \
                            x[:, :-n1 or None, :-n2 or None, :-n3 or None, :]

        return tf.reduce_mean((1.0 / math.sqrt(math.pow(n1, 2.0) * math.pow(n2, 2.0) * math.pow(n3, 2.0))) *
                              (tf.math.pow(current_pixel_dif, 2.0) /
                               ((x[:, n1 or None:, n2 or None:, n3 or None:, :] +
                                 x[:, :-n1 or None, :-n2 or None, :-n3 or None, :]) +
                                (_gamma * tf.math.abs(current_pixel_dif)))))

    def _relative_difference_loss(x, n, _gamma, _pixel_dif):
        _pixel_dif.append(_pixel_dif_negative(x, n, 0, 0, _gamma))
        _pixel_dif.append(_pixel_dif_negative(x, 0, n, 0, _gamma))
        _pixel_dif.append(_pixel_dif_negative(x, 0, 0, n, _gamma))

        _pixel_dif.append(_pixel_dif_negative(x, n, n, 0, _gamma))
        _pixel_dif.append(_pixel_dif_negative(x, n, 0, n, _gamma))
        _pixel_dif.append(_pixel_dif_negative(x, 0, n, n, _gamma))

        _pixel_dif.append(_pixel_dif_negative(x, n, n, n, _gamma))

        _pixel_dif.append(_pixel_dif_positive(x, n, 0, 0, _gamma))
        _pixel_dif.append(_pixel_dif_positive(x, 0, n, 0, _gamma))
        _pixel_dif.append(_pixel_dif_positive(x, 0, 0, n, _gamma))

        _pixel_dif.append(_pixel_dif_positive(x, n, n, 0, _gamma))
        _pixel_dif.append(_pixel_dif_positive(x, n, 0, n, _gamma))
        _pixel_dif.append(_pixel_dif_positive(x, 0, n, n, _gamma))

        _pixel_dif.append(_pixel_dif_positive(x, n, n, n, _gamma))

        return _pixel_dif

    y_pred = tf.cast(y_pred, dtype=tf.float64)

    # The input is a batch of images with shape:
    # [batch, height, width, depth, channels].

    y_pred = y_pred - tf.math.reduce_min(y_pred)
    y_pred = tf.pad(y_pred, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")

    gamma = 0.0

    pixel_dif = []

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height, width and depth by slicing.
    # Calculate the total variation by taking the absolute value of the
    # pixel-differences
    pixel_dif = _relative_difference_loss(y_pred, 1, gamma, pixel_dif)

    # summing over the appropriate axis.
    return tf.reduce_mean(tf.convert_to_tensor(pixel_dif))


def log_cosh_relative_difference_loss(y_true, y_pred):
    return ((1.0 * log_cosh_loss(y_true, y_pred)) + (1e-01 * relative_difference_loss(y_true, y_pred))) / 2.0


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
    return (correlation_coefficient_loss(y_true, y_pred) * -1.0) + 1.0
