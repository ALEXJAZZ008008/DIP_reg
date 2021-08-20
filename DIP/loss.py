# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf
import tensorflow.keras as k


# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred

    mx = k.backend.mean(x)
    my = k.backend.mean(y)

    xm = x - mx
    ym = y - my

    r_num = k.backend.sum(tf.multiply(xm, ym))
    r_den = k.backend.sqrt(tf.multiply(k.backend.sum(k.backend.square(xm)), k.backend.sum(k.backend.square(ym))))
    r = r_num / r_den

    r = k.backend.maximum(k.backend.minimum(r, 1.0), -1.0)

    return 1 - k.backend.square(r)


def accuracy_correlation_coefficient(y_true, y_pred):
    return (correlation_coefficient_loss(y_true, y_pred) * -1.0) + 1.0
