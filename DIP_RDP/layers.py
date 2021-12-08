# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as k


import main

if main.reproducible_bool:
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(main.seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(main.seed_value)


import losses
import parameters


def get_gaussian_noise(x, sigma):
    print("get_gaussian_noise")

    if sigma > 0.0:
        x = k.layers.GaussianNoise(sigma)(x)

    return x


# https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/#reflection-padding
'''
  3D Reflection Padding
  Attributes:
    - padding: (padding_width, padding_height) tuple
'''


class ReflectionPadding3D(k.layers.Layer):
    def __init__(self, padding=(0, 0, 0), **kwargs):
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
                                     [0, 0]], "REFLECT")


def get_reflection_padding(x, size):
    print("get_reflection_padding")

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


def get_channel_shuffle(x, groups):
    print("get_channel_shuffle")

    if groups > 1:
        x = k.layers.Lambda(channel_shuffle, arguments={"groups": groups})(x)

    return x


# https://github.com/keras-team/keras/blob/v2.6.0/keras/layers/core.py#L1274-L1303
class ActivityRegularization(k.layers.Layer):
    """Layer that applies an update to the cost function based input activity.

    Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

    Output shape:
    Same shape as input.
    """

    def __init__(self, **kwargs):
        super(ActivityRegularization, self).__init__(activity_regularizer=losses.log_cosh_regulariser, **kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(ActivityRegularization, self).get_config()

        return dict(list(base_config.items()))


def get_dropout(x):
    print("get_dropout")

    if parameters.dropout > 0.0:
        if len(x.shape) > 2:
            x = k.layers.SpatialDropout3D(rate=parameters.dropout)(x)
        else:
            x = k.layers.Dropout(rate=parameters.dropout)(x)

    return x


def get_convolution_layer(x, depth, size, stride, groups):
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
                        kernel_regularizer=losses.l2_regulariser)(x)
    x = get_channel_shuffle(x, groups)
    x = tfa.layers.GroupNormalization(groups=groups)(x)  # noqa
    x = get_gaussian_noise(x, parameters.layer_gaussian_sigma)
    x = k.layers.PReLU(alpha_initializer=k.initializers.Constant(1.0),
                       alpha_regularizer=losses.log_cosh_regulariser,
                       shared_axes=[1, 2, 3])(x)
    x = ActivityRegularization()(x)
    x = get_dropout(x)

    return x


def get_transpose_convolution_layer(x, depth, size, stride, groups):
    print("get_transpose_convolution_layer")

    x = k.layers.Convolution3DTranspose(filters=depth,
                                        kernel_size=size,
                                        strides=stride,
                                        dilation_rate=(1, 1, 1),
                                        groups=groups,
                                        padding="same",
                                        kernel_initializer="he_normal",
                                        bias_initializer=k.initializers.Constant(0.0),
                                        kernel_regularizer=losses.l2_regulariser)(x)
    x = get_channel_shuffle(x, groups)
    x = tfa.layers.GroupNormalization(groups=groups)(x)  # noqa
    x = get_gaussian_noise(x, parameters.layer_gaussian_sigma)
    x = k.layers.PReLU(alpha_initializer=k.initializers.Constant(1.0),
                       alpha_regularizer=losses.log_cosh_regulariser,
                       shared_axes=[1, 2, 3])(x)
    x = ActivityRegularization()(x)
    x = get_dropout(x)

    return x
