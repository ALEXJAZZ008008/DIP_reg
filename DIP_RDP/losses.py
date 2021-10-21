# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf


import main

if main.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(main.seed_value)


import parameters


def log_cosh_loss(y_true, y_pred):
    def _log_cosh(x):
        return (x + tf.math.softplus(-2.0 * x)) - tf.math.log(2.0)

    return tf.math.reduce_mean(_log_cosh(y_pred - y_true))


def total_variation_loss(y_true, y_pred):
    def _pixel_dif_one_distance(n):
        return 1.0 / tf.cast(n, dtype=tf.float32)

    def _pixel_dif_one_1(x, n):
        return _pixel_dif_one_distance(n) * \
               tf.math.reduce_mean(log_cosh_loss(x[:, n:, :, :, :], x[:, :-n, :, :, :]))

    def _pixel_dif_one_2(x, n):
        return _pixel_dif_one_distance(n) * \
               tf.math.reduce_mean(log_cosh_loss(x[:, :, n:, :, :], x[:, :, :-n, :, :]))

    def _pixel_dif_one_3(x, n):
        return _pixel_dif_one_distance(n) * \
               tf.math.reduce_mean(log_cosh_loss(x[:, :, :, n:, :], x[:, :, :, :-n, :]))

    def _pixel_dif_one(x, n):
        return tf.math.reduce_sum(tf.stack([_pixel_dif_one_1(x, n),
                                            _pixel_dif_one_2(x, n),
                                            _pixel_dif_one_3(x, n)]))

    def _pixel_dif_two_distance(n1, n2):
        return 1.0 / tf.math.sqrt(tf.math.square(tf.cast(n1, dtype=tf.float32)) +
                                  tf.math.square(tf.cast(n2, dtype=tf.float32)))

    def _pixel_dif_two_1(x, n1, n2):
        return _pixel_dif_two_distance(n1, n2) * \
               tf.math.reduce_mean(log_cosh_loss(x[:, n1:, n2:, :, :], x[:, :-n1, :-n2, :, :]))

    def _pixel_dif_two_2(x, n1, n2):
        return _pixel_dif_two_distance(n1, n2) * \
               tf.math.reduce_mean(log_cosh_loss(x[:, n1:, :, n2:, :], x[:, :-n1, :, :-n2, :]))

    def _pixel_dif_two_3(x, n1, n2):
        return _pixel_dif_two_distance(n1, n2) * \
               tf.math.reduce_mean(log_cosh_loss(x[:, :, n1:, n2:, :], x[:, :, :-n1, :-n2, :]))

    def _pixel_dif_two(x, n1, n2):
        return tf.math.reduce_sum(tf.stack([_pixel_dif_two_1(x, n1, n2),
                                            _pixel_dif_two_2(x, n1, n2),
                                            _pixel_dif_two_3(x, n1, n2)]))

    def _pixel_dif_three_distance(n1, n2, n3):
        return 1.0 / tf.math.sqrt(tf.math.square(tf.cast(n1, dtype=tf.float32)) +
                                  tf.math.square(tf.cast(n2, dtype=tf.float32)) +
                                  tf.math.square(tf.cast(n3, dtype=tf.float32)))

    def _pixel_dif_three_1(x, n1, n2, n3):
        return _pixel_dif_three_distance(n1, n2, n3) * \
               tf.math.reduce_mean(log_cosh_loss(x[:, n1:, n2:, n3:, :], x[:, :-n1, :-n2, :-n3, :]))

    def _pixel_dif_three(x, n1, n2, n3):
        return tf.math.reduce_sum(tf.stack([_pixel_dif_three_1(x, n1, n2, n3)]))

    y_pred = y_pred - tf.math.reduce_min(y_pred)

    # The input is a batch of images with shape:
    # [batch, height, width, depth, channels].

    pixel_dif = 0.0

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height, width and depth by slicing.
    # Calculate the total variation by taking the absolute value of the
    # pixel-differences summing over the appropriate axis.
    pixel_dif = pixel_dif + _pixel_dif_one(y_pred, 1)
    pixel_dif = pixel_dif + _pixel_dif_two(y_pred, 1, 1)
    pixel_dif = pixel_dif + _pixel_dif_three(y_pred, 1, 1, 1)

    return pixel_dif / 7.0


def log_cosh_total_variation_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return log_cosh_loss(y_true, y_pred) + \
           (parameters.total_variation_weight * relative_difference_loss(y_true, y_pred))


def relative_difference_loss(y_true, y_pred):
    def _pixel_dif_one_distance(n):
        return 1.0 / tf.cast(n, dtype=tf.float32)

    def _pixel_dif_one_1(x, n, _gamma):
        def _pixel_dif(_x, current_pixel_dif, _n, __gamma):
            return _pixel_dif_one_distance(_n) * \
                   tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.square(current_pixel_dif),
                                       ((_x[:, _n:, :, :, :] + _x[:, :-_n, :, :, :]) +
                                        (__gamma * tf.math.abs(current_pixel_dif)))))

        return _pixel_dif(x, x[:, n:, :, :, :] - x[:, :-n, :, :, :], n, _gamma)

    def _pixel_dif_one_2(x, n, _gamma):
        def _pixel_dif(_x, current_pixel_dif, _n, __gamma):
            return _pixel_dif_one_distance(_n) * \
                   tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.square(current_pixel_dif),
                                       ((_x[:, :, _n:, :, :] + _x[:, :, :-_n, :, :]) +
                                        (__gamma * tf.math.abs(current_pixel_dif)))))

        return _pixel_dif(x, x[:, :, n:, :, :] - x[:, :, :-n, :, :], n, _gamma)

    def _pixel_dif_one_3(x, n, _gamma):
        def _pixel_dif(_x, current_pixel_dif, _n, __gamma):
            return _pixel_dif_one_distance(_n) * \
                   tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.square(current_pixel_dif),
                                       ((_x[:, :, :, _n:, :] + _x[:, :, :, :-_n, :]) +
                                        (__gamma * tf.math.abs(current_pixel_dif)))))

        return _pixel_dif(x, x[:, :, :, n:, :] - x[:, :, :, :-n, :], n, _gamma)

    def _pixel_dif_one(x, n, _gamma):
        return tf.math.reduce_sum(tf.stack([_pixel_dif_one_1(x, n, _gamma),
                                            _pixel_dif_one_2(x, n, _gamma),
                                            _pixel_dif_one_3(x, n, _gamma)]))

    def _pixel_dif_two_distance(n1, n2):
        return 1.0 / tf.math.sqrt(tf.math.square(tf.cast(n1, dtype=tf.float32)) +
                                  tf.math.square(tf.cast(n2, dtype=tf.float32)))

    def _pixel_dif_two_1(x, n1, n2, _gamma):
        def _pixel_dif(_x, current_pixel_dif, _n1, _n2, __gamma):
            return _pixel_dif_two_distance(_n1, _n2) * \
                   tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.square(current_pixel_dif),
                                       ((_x[:, _n1:, _n2:, :, :] + _x[:, :-_n1, :-_n2, :, :]) +
                                        (__gamma * tf.math.abs(current_pixel_dif)))))

        return _pixel_dif(x, x[:, n1:, n2:, :, :] - x[:, :-n1, :-n2, :, :], n1, n2, _gamma)

    def _pixel_dif_two_2(x, n1, n2, _gamma):
        def _pixel_dif(_x, current_pixel_dif, _n1, _n2, __gamma):
            return _pixel_dif_two_distance(_n1, _n2) * \
                   tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.square(current_pixel_dif),
                                       ((_x[:, _n1:, :, _n2:, :] + _x[:, :-_n1, :, :-_n2, :]) +
                                        (__gamma * tf.math.abs(current_pixel_dif)))))

        return _pixel_dif(x, x[:, n1:, :, n2:, :] - x[:, :-n1, :, :-n2, :], n1, n2, _gamma)

    def _pixel_dif_two_3(x, n1, n2, _gamma):
        def _pixel_dif(_x, current_pixel_dif, _n1, _n2, __gamma):
            return _pixel_dif_two_distance(_n1, _n2) * \
                   tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.square(current_pixel_dif),
                                       ((_x[:, :, _n1:, _n2:, :] + _x[:, :, :-_n1, :-_n2, :]) +
                                        (__gamma * tf.math.abs(current_pixel_dif)))))

        return _pixel_dif(x, x[:, :, n1:, n2:, :] - x[:, :, :-n1, :-n2, :], n1, n2, _gamma)

    def _pixel_dif_two(x, n1, n2, _gamma):
        return tf.math.reduce_sum(tf.stack([_pixel_dif_two_1(x, n1, n2, _gamma),
                                            _pixel_dif_two_2(x, n1, n2, _gamma),
                                            _pixel_dif_two_3(x, n1, n2, _gamma)]))

    def _pixel_dif_three_distance(n1, n2, n3):
        return 1.0 / tf.math.sqrt(tf.math.square(tf.cast(n1, dtype=tf.float32)) +
                                  tf.math.square(tf.cast(n2, dtype=tf.float32)) +
                                  tf.math.square(tf.cast(n3, dtype=tf.float32)))

    def _pixel_dif_three_1(x, n1, n2, n3, _gamma):
        def _pixel_dif(_x, current_pixel_dif, _n1, _n2, _n3, __gamma):
            return _pixel_dif_three_distance(_n1, _n2, _n3) * \
                   tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.square(current_pixel_dif),
                                       ((_x[:, _n1:, _n2:, _n3:, :] + _x[:, :-_n1, :-_n2, :-_n3, :]) +
                                        (__gamma * tf.math.abs(current_pixel_dif)))))

        return _pixel_dif(x, x[:, n1:, n2:, n3:, :] - x[:, :-n1, :-n2, :-n3, :], n1, n2, n3, _gamma)

    def _pixel_dif_three(x, n1, n2, n3, _gamma):
        return tf.math.reduce_sum(tf.stack([_pixel_dif_three_1(x, n1, n2, n3, _gamma)]))

    y_pred = y_pred - tf.math.reduce_min(y_pred)

    # The input is a batch of images with shape:
    # [batch, height, width, depth, channels].

    gamma = parameters.relative_difference_edge_preservation_weight

    pixel_dif = 0.0

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height, width and depth by slicing.
    # Calculate the total variation by taking the absolute value of the
    # pixel-differences summing over the appropriate axis.
    pixel_dif = pixel_dif + _pixel_dif_one(y_pred, 1, gamma)
    pixel_dif = pixel_dif + _pixel_dif_two(y_pred, 1, 1, gamma)
    pixel_dif = pixel_dif + _pixel_dif_three(y_pred, 1, 1, 1, gamma)

    return pixel_dif / 7.0


def log_cosh_relative_difference_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return log_cosh_loss(y_true, y_pred) + \
           (parameters.relative_difference_weight * relative_difference_loss(y_true, y_pred))


def max_min_constraint(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return parameters.max_min_constraint_weight * \
           (tf.math.abs(tf.math.maximum(tf.math.reduce_max(y_true), tf.math.reduce_max(y_pred))) +
            tf.math.abs(tf.math.minimum(tf.math.reduce_min(y_true), tf.math.reduce_min(y_pred))))


def log_cosh_regulariser(weight_matrix):
    def _log_cosh(x):
        return (x + tf.math.softplus(-2.0 * x)) - tf.math.log(2.0)

    weight_matrix = tf.cast(weight_matrix, dtype=tf.float32)

    return tf.math.reduce_mean(_log_cosh(weight_matrix))


def l2_regulariser(weight_matrix):
    weight_matrix = tf.cast(weight_matrix, dtype=tf.float32)

    return parameters.l2_weight * tf.math.reduce_mean(tf.math.square(weight_matrix))


def correlation_coefficient_loss(y_true, y_pred):
    def _correlation_coefficient(xm, ym):
        return 1.0 - tf.math.square(tf.math.maximum(
            tf.math.minimum(
                tf.math.reduce_sum((xm * ym)) /
                tf.math.sqrt(tf.math.reduce_sum(tf.math.square(xm)) * tf.math.reduce_sum(tf.math.square(ym))),
                1.0),
            -1.0))

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return _correlation_coefficient(y_true - tf.math.reduce_mean(y_true), y_pred - tf.math.reduce_mean(y_pred))


def correlation_coefficient_accuracy(y_true, y_pred):
    return (correlation_coefficient_loss(y_true, y_pred) * -1.0) + 1.0
