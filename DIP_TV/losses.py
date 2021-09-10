# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf
import tensorflow.keras as k


import main


if main.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(main.seed_value)


# https://github.com/keras-team/keras/blob/master/keras/losses.py#L256-L310
def mean_squared_error_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    return k.backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)


# https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/python/ops/image_ops_impl.py#L3213-L3282
def total_variation(images):
    # The input is a batch of images with shape:
    # [batch, height, width, depth, channels].

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height, width and depth by slicing.
    pixel_dif1 = images[:, 1:, :, :, :] - images[:, :-1, :, :, :]
    pixel_dif2 = images[:, :, 1:, :, :] - images[:, :, :-1, :, :]
    pixel_dif3 = images[:, :, :, 1:, :] - images[:, :, :, :-1, :]

    # Only sum for the last 4 axis.
    # This results in a 1-D tensor with the total variation for each image.
    sum_axis = [1, 2, 3, 4]

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (tf.math.reduce_sum(tf.math.abs(pixel_dif1), axis=sum_axis) +
               tf.math.reduce_sum(tf.math.abs(pixel_dif2), axis=sum_axis) +
               tf.math.reduce_sum(tf.math.abs(pixel_dif3), axis=sum_axis))

    return tot_var


def total_variation_loss(_, y_pred):
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    return tf.reduce_sum(total_variation(y_pred))


def mean_square_error_total_variation_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    return ((1.0 * mean_squared_error_loss(y_true, y_pred)) + (9e-08 * total_variation_loss(y_true, y_pred))) / 2.0


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
