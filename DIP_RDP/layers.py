# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import numpy as np
import tensorflow as tf
import tensorflow.keras as k


# https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/#reflection-padding
'''
  3D Reflection Padding
  Attributes:
    - padding: (padding_width, padding_height) tuple
'''


class ReflectionPadding3D(k.layers.Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + (2 * self.padding[0]),
                input_shape[2] + (2 * self.padding[1]),
                input_shape[3] + (2 * self.padding[2]),
                input_shape[4])

    def call(self, input_tensor, *args, **kwargs):
        return tf.pad(input_tensor, [[0, 0],
                                     [self.padding[0], self.padding[0]],
                                     [self.padding[1], self.padding[1]],
                                     [self.padding[2], self.padding[2]],
                                     [0, 0]], 'REFLECT')


def get_reflection_padding(x, size):
    if size[0] % 2 > 0:
        kernel_padding_1 = int(np.floor(size[0] / 2))
    else:
        kernel_padding_1 = int(size[0] / 2) - 1

    if size[1] % 2 > 0:
        kernel_padding_2 = int(np.floor(size[1] / 2))
    else:
        kernel_padding_2 = int(size[1] / 2) - 1

    if size[2] % 2 > 0:
        kernel_padding_3 = int(np.floor(size[2] / 2))
    else:
        kernel_padding_3 = int(size[2] / 2) - 1

    if kernel_padding_1 != 0 or kernel_padding_2 != 0 or kernel_padding_3 != 0:
        x = ReflectionPadding3D((kernel_padding_1, kernel_padding_2, kernel_padding_3))(x)

    return x


# https://github.com/scheckmedia/keras-shufflenet/blob/master/shufflenet.py
def channel_shuffle(x, groups):
    """
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    groups: int
        number of groups per channel
    Returns
    -------
        channel shuffled output tensor
    Examples
    --------
    Example for a 1D Array with 3 groups
    d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    x = np.reshape(d, (3, 3))
    x = np.transpose(x, [1, 0])
    x = np.reshape(x, (9,))
    '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'
    """
    height, width, depth, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = k.backend.reshape(x, [-1, height, width, depth, groups, channels_per_group])
    x = k.backend.permute_dimensions(x, (0, 1, 2, 3, 5, 4))  # transpose
    x = k.backend.reshape(x, [-1, height, width, depth, in_channels])

    return x


def get_convolution_layer(x, depth, size, stride, groups, kernel_weight, activity_sparseness):
    print("get_convolution_layer")

    x = get_reflection_padding(x, size)
    x = k.layers.Conv3D(filters=depth,
                        kernel_size=size,
                        strides=stride,
                        dilation_rate=(1, 1, 1),
                        groups=groups,
                        padding="valid",
                        kernel_initializer="he_normal",
                        bias_initializer=k.initializers.Constant(0.0),
                        kernel_regularizer=k.regularizers.l2(l2=kernel_weight),
                        activity_regularizer=k.regularizers.l1(l1=activity_sparseness))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.ReLU(negative_slope=0.2)(x)
    x = k.layers.Lambda(channel_shuffle, arguments={"groups": groups})(x)

    return x


def get_transpose_convolution_layer(x, depth, size, stride, groups, kernel_weight, activity_sparseness):
    print("get_transpose_convolution_layer")

    x = k.layers.Conv3DTranspose(filters=depth,
                                 kernel_size=size,
                                 strides=stride,
                                 dilation_rate=(1, 1, 1),
                                 groups=groups,
                                 padding="same",
                                 kernel_initializer="he_normal",
                                 bias_initializer=k.initializers.Constant(0.0),
                                 kernel_regularizer=k.regularizers.l2(l2=kernel_weight),
                                 activity_regularizer=k.regularizers.l1(l1=activity_sparseness))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.ReLU(negative_slope=0.2)(x)
    x = k.layers.Lambda(channel_shuffle, arguments={"groups": groups})(x)

    return x
