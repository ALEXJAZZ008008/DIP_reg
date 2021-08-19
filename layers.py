# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import numpy as np
import tensorflow as tf
import tensorflow.keras as k

import parameters


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
    >>> d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> x = np.reshape(d, (3, 3))
    >>> x = np.transpose(x, [1, 0])
    >>> x = np.reshape(x, (9,))
    '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'
    """
    height, width, depth, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = k.backend.reshape(x, [-1, height, width, depth, groups, channels_per_group])
    x = k.backend.permute_dimensions(x, (0, 1, 2, 3, 5, 4))  # transpose
    x = k.backend.reshape(x, [-1, height, width, depth, in_channels])

    return x


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
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3] + 2 * self.padding[2],
                input_shape[4])

    def call(self, input_tensor, *args, **kwargs):
        padding_depth, padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0, 0],
                                     [padding_height, padding_height],
                                     [padding_width, padding_width],
                                     [padding_depth, padding_depth],
                                     [0, 0]], 'REFLECT')


def get_kernel_groups(x, grouped_bool, depth):
    if grouped_bool and x.shape[-1] > 1 and depth > 2:
        kernel_groups = 2

        if x.shape[-1] % kernel_groups > 0 or depth % kernel_groups > 0:
            kernel_groups = 1
    else:
        kernel_groups = 1

    return kernel_groups


def bottleneck_conv3D(x, depth, grouped_bool, grouped_channel_shuffle_bool, kernel_groups, train_bool, ltwo_weight,
                      lone_weight, linear_bool, gaussian_stddev, dropout_amount):
    print("bottleneck_conv3D")

    if grouped_bool and grouped_channel_shuffle_bool:
        x = k.layers.TimeDistributed(k.layers.Lambda(channel_shuffle,
                                                     arguments={"groups": kernel_groups}),
                                     trainable=train_bool)(x)

    x = k.layers.TimeDistributed(k.layers.Conv3D(filters=depth,
                                                 kernel_size=(1, 1, 1),
                                                 strides=(1, 1, 1),
                                                 dilation_rate=(1, 1, 1),
                                                 groups=kernel_groups,
                                                 padding="valid",
                                                 kernel_initializer="he_uniform",
                                                 bias_initializer=k.initializers.Constant(0.0),
                                                 kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                 activity_regularizer=k.regularizers.l1(l1=lone_weight),
                                                 kernel_constraint=k.constraints.UnitNorm(),
                                                 trainable=train_bool),
                                 trainable=train_bool)(x)
    x = k.layers.BatchNormalization(trainable=train_bool)(x)

    if not linear_bool:
        if gaussian_stddev > 0.0:
            x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                       trainable=train_bool)(x)

        x = k.layers.ReLU(max_value=6.0,
                          negative_slope=0.2)(x)

    if x.shape[-1] > 1 and dropout_amount > 0.0:
        x = k.layers.TimeDistributed(k.layers.SpatialDropout3D(dropout_amount,
                                                               trainable=train_bool),
                                     trainable=train_bool)(x)

    return x



def grouped_depthwise_conv3D(x, grouped_bool, grouped_channel_shuffle_bool, kernel_padding, kernel_size, kernel_stride,
                             kernel_dilation, ltwo_weight, lone_weight, linear_bool, gaussian_stddev, dropout_amount,
                             train_bool):
    print("grouped_depthwise_conv3D")

    if grouped_bool and grouped_channel_shuffle_bool:
        x = k.layers.TimeDistributed(k.layers.Lambda(channel_shuffle,
                                                     arguments={"groups": x.shape[-1]}),
                                     trainable=train_bool)(x)

    if kernel_padding[0] > 0 or kernel_padding[1] > 0 or kernel_padding[2] > 0:
        x = k.layers.TimeDistributed(ReflectionPadding3D(padding=kernel_padding,
                                                         trainable=train_bool),
                                     trainable=train_bool)(x)

    x = k.layers.TimeDistributed(k.layers.Conv3D(filters=x.shape[-1],
                                                 kernel_size=kernel_size,
                                                 strides=kernel_stride,
                                                 dilation_rate=kernel_dilation,
                                                 groups=x.shape[-1],
                                                 padding="valid",
                                                 kernel_initializer="he_uniform",
                                                 bias_initializer=k.initializers.Constant(0.0),
                                                 kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                 activity_regularizer=k.regularizers.l1(l1=lone_weight),
                                                 kernel_constraint=k.constraints.UnitNorm(),
                                                 trainable=train_bool),
                                 trainable=train_bool)(x)
    x = k.layers.BatchNormalization(trainable=train_bool)(x)

    if not linear_bool:
        if gaussian_stddev > 0.0:
            x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                       trainable=train_bool)(x)

        x = k.layers.ReLU(max_value=6.0,
                          negative_slope=0.2)(x)

    if x.shape[-1] > 1 and dropout_amount > 0.0:
        x = k.layers.TimeDistributed(k.layers.SpatialDropout3D(dropout_amount,
                                                               trainable=train_bool),
                                     trainable=train_bool)(x)

    return x


def conv3D_scaling(x, stride, dilated_bool, size, depth, grouped_bool, convolved_fov_bool, bottleneck_bool):
    if stride[0] > 1 or stride[1] > 1 or stride[2] > 1:
        convolved_fov_bool = False

        if dilated_bool:
            size = (3, 3, 3)

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

    kernel_padding = (kernel_padding_1, kernel_padding_2, kernel_padding_3)

    kernel_groups = get_kernel_groups(x, grouped_bool, depth)

    if convolved_fov_bool:
        fov_loops = int(np.floor(np.max(size) / 2))
        kernel_size = (3, 3, 3)
        kernel_dilation = (1, 1, 1)
    else:
        fov_loops = 1
        kernel_size = size

        if dilated_bool:
            kernel_dilation = (int(np.floor(size[0] / 2)), int(np.floor(size[1] / 2)), int(np.floor(size[2] / 2)))
            kernel_size = (3, 3, 3)
        else:
            kernel_dilation = (1, 1, 1)

    if bottleneck_bool:
        kernel_depth = depth * parameters.bottleneck_expand_multiplier
    else:
        kernel_depth = depth

    return convolved_fov_bool, kernel_padding, kernel_groups, fov_loops, kernel_size, kernel_dilation, kernel_depth


def get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, depth, size, stride, ltwo_weight, lone_weight,
                          gaussian_stddev, dropout_amount, resnet_bool, train_bool, concatenate_bool, densenet_bool):
    print("get_convolution_layer")

    convolved_fov_bool = False
    dilated_bool = True
    depthwise_seperable_bool = parameters.depthwise_seperable_bool
    bottleneck_bool = parameters.bottleneck_bool

    convolved_fov_bool, kernel_padding, kernel_groups, fov_loops, kernel_size, kernel_dilation, kernel_depth = \
        conv3D_scaling(x, stride, dilated_bool, size, depth, grouped_bool, convolved_fov_bool, bottleneck_bool)

    res_connections = []

    for i in range(fov_loops):
        if i >= fov_loops - 1:
            kernel_stride = stride
        else:
            kernel_stride = (1, 1, 1)

        if resnet_bool:
            res_connections.append(x)

        if bottleneck_bool:
            x = bottleneck_conv3D(x, depth, grouped_bool, grouped_channel_shuffle_bool, kernel_groups, train_bool,
                                  ltwo_weight, lone_weight, False, gaussian_stddev, dropout_amount)

        if depthwise_seperable_bool:
            x = grouped_depthwise_conv3D(x, grouped_bool, grouped_channel_shuffle_bool, kernel_padding, kernel_size,
                                         kernel_stride, kernel_dilation, ltwo_weight, lone_weight, False,
                                         gaussian_stddev, dropout_amount, train_bool)

            if grouped_bool and grouped_channel_shuffle_bool:
                x = k.layers.TimeDistributed(k.layers.Lambda(channel_shuffle,
                                                             arguments={"groups": kernel_groups}),
                                             trainable=train_bool)(x)

            x = k.layers.TimeDistributed(k.layers.Conv3D(filters=kernel_depth,
                                                         kernel_size=(1, 1, 1),
                                                         strides=(1, 1, 1),
                                                         dilation_rate=(1, 1, 1),
                                                         groups=kernel_groups,
                                                         padding="valid",
                                                         kernel_initializer="he_uniform",
                                                         bias_initializer=k.initializers.Constant(0.0),
                                                         kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                         activity_regularizer=k.regularizers.l1(l1=lone_weight),
                                                         kernel_constraint=k.constraints.UnitNorm(),
                                                         trainable=train_bool),
                                         trainable=train_bool)(x)
            x = k.layers.BatchNormalization(trainable=train_bool)(x)

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=train_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            if x.shape[-1] > 1 and dropout_amount > 0.0:
                x = k.layers.TimeDistributed(k.layers.SpatialDropout3D(dropout_amount,
                                                                       trainable=train_bool),
                                             trainable=train_bool)(x)
        else:
            if grouped_bool and grouped_channel_shuffle_bool:
                x = k.layers.TimeDistributed(k.layers.Lambda(channel_shuffle,
                                                             arguments={"groups": kernel_groups}),
                                             trainable=train_bool)(x)

            if kernel_padding[0] > 0 or kernel_padding[1] > 0 or kernel_padding[2] > 0:
                x = k.layers.TimeDistributed(ReflectionPadding3D(padding=kernel_padding,
                                                                 trainable=train_bool),
                                             trainable=train_bool)(x)

            x = k.layers.TimeDistributed(k.layers.Conv3D(filters=kernel_depth,
                                                         kernel_size=kernel_size,
                                                         strides=kernel_stride,
                                                         dilation_rate=kernel_dilation,
                                                         groups=kernel_groups,
                                                         padding="valid",
                                                         kernel_initializer="he_uniform",
                                                         bias_initializer=k.initializers.Constant(0.0),
                                                         kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                         activity_regularizer=k.regularizers.l1(l1=lone_weight),
                                                         kernel_constraint=k.constraints.UnitNorm(),
                                                         trainable=train_bool),
                                         trainable=train_bool)(x)
            x = k.layers.BatchNormalization(trainable=train_bool)(x)

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=train_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            if x.shape[-1] > 1 and dropout_amount > 0.0:
                x = k.layers.TimeDistributed(k.layers.SpatialDropout3D(dropout_amount,
                                                                       trainable=train_bool),
                                             trainable=train_bool)(x)

        if bottleneck_bool:
            x = bottleneck_conv3D(x, depth, grouped_bool, grouped_channel_shuffle_bool, kernel_groups, train_bool,
                                  ltwo_weight, lone_weight, True, gaussian_stddev, dropout_amount)

        if resnet_bool and not (stride[0] > 1 or stride[1] > 1 or stride[2] > 1):
            if concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=train_bool)([x] + res_connections)

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=train_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = bottleneck_conv3D(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                      kernel_groups, train_bool, ltwo_weight, lone_weight, False,
                                      gaussian_stddev, dropout_amount)
            else:
                for j in range(len(res_connections)):
                    if res_connections[j].shape[-1] != x.shape[-1]:
                        kernel_groups = get_kernel_groups(res_connections[j], grouped_bool, x.shape[-1])

                        res_connections[j] = bottleneck_conv3D(res_connections[j], x.shape[-1], grouped_bool,
                                                               grouped_channel_shuffle_bool, kernel_groups, train_bool,
                                                               ltwo_weight, lone_weight, False, gaussian_stddev,
                                                               dropout_amount)

                x = k.layers.Add(trainable=train_bool)([x] + res_connections)

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=train_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

            if not densenet_bool:
                res_connections = []
            else:
                res_connections = []

    return x


# https://www.kaggle.com/qqgeogor/keras-GRU-attention-glove840b-lb-0-043#L51
class Attention(k.layers.Layer):
    def __init__(self, step_dim, w_regularizer=None, b_regularizer=None, w_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            3D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/GRU/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(GRU(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = k.initializers.get('glorot_uniform')

        self.W_regularizer = k.regularizers.get(w_regularizer)
        self.b_regularizer = k.regularizers.get(b_regularizer)

        self.W_constraint = k.constraints.get(w_constraint)
        self.b_constraint = k.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        self.W = None
        self.b = None

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],), initializer=self.init, name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer, constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],), initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer, constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None, *args, **kwargs):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = k.backend.reshape(k.backend.dot(k.backend.reshape(x, (-1, features_dim)),
                                              k.backend.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij = eij + self.b

        eij = k.backend.tanh(eij)

        a = k.backend.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a = a * k.backend.cast(mask, k.backend.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a = a / k.backend.cast(k.backend.sum(a, axis=1, keepdims=True) + k.backend.epsilon(), k.backend.floatx())

        a = k.backend.expand_dims(a)
        weighted_input = x * a

        return k.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/
# Define sampling with reparameterization trick
def sample_x(args):
    mu, sigma = args

    mu = tf.cast(mu, dtype=tf.float32)
    sigma = tf.cast(sigma, dtype=tf.float32)

    batch = k.backend.shape(mu)[0]
    dim = k.backend.int_shape(mu)[1]

    eps = k.backend.random_normal((batch, dim))

    return mu + k.backend.exp(sigma / 2.0) * eps


def get_dv_latent_layer(x, latent_size, train_bool):
    print("get_dv_latent_layer")

    mu = k.layers.Bidirectional(k.layers.GRU(units=latent_size,
                                             activation="linear",
                                             recurrent_activation="sigmoid",
                                             use_bias=True,
                                             kernel_initializer="glorot_uniform",
                                             recurrent_initializer="glorot_uniform",
                                             bias_initializer=k.initializers.Constant(0.0),
                                             kernel_regularizer=None,
                                             activity_regularizer=None,
                                             kernel_constraint=None,
                                             dropout=0.0,
                                             return_sequences=True),
                                merge_mode="sum",
                                trainable=train_bool,
                                name="latent_mu")(x)

    sigma = k.layers.Bidirectional(k.layers.GRU(units=latent_size,
                                                activation="linear",
                                                recurrent_activation="sigmoid",
                                                use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="glorot_uniform",
                                                bias_initializer=k.initializers.Constant(0.0),
                                                kernel_regularizer=None,
                                                activity_regularizer=None,
                                                kernel_constraint=None,
                                                dropout=0.0,
                                                return_sequences=True),
                                   merge_mode="sum",
                                   trainable=train_bool,
                                   name="latent_sigma")(x)

    # Use reparameterization trick to ensure correct gradient
    x = k.layers.Lambda(sample_x,
                        output_shape=(latent_size,),
                        trainable=train_bool,
                        name="latent")([mu, sigma])

    return x, mu, sigma


def bottleneck_conv3D_transpose(x, depth, grouped_bool, grouped_channel_shuffle_bool, kernel_groups, train_bool,
                                ltwo_weight, lone_weight, linear_bool, gaussian_stddev, dropout_amount):
    print("linear_bottleneck_conv3D_transpose")

    if grouped_bool and grouped_channel_shuffle_bool:
        x = k.layers.TimeDistributed(k.layers.Lambda(channel_shuffle,
                                                     arguments={"groups": kernel_groups}),
                                     trainable=train_bool)(x)

    x = k.layers.TimeDistributed(k.layers.Conv3DTranspose(filters=depth,
                                                          kernel_size=(1, 1, 1),
                                                          strides=(1, 1, 1),
                                                          dilation_rate=(1, 1, 1),
                                                          groups=kernel_groups,
                                                          padding="same",
                                                          kernel_initializer="he_uniform",
                                                          bias_initializer=k.initializers.Constant(0.0),
                                                          kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                          activity_regularizer=
                                                          k.regularizers.l1(l1=lone_weight),
                                                          kernel_constraint=k.constraints.UnitNorm(),
                                                          trainable=train_bool),
                                 trainable=train_bool)(x)
    x = k.layers.BatchNormalization(trainable=train_bool)(x)

    if not linear_bool:
        if gaussian_stddev > 0.0:
            x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                       trainable=train_bool)(x)

        x = k.layers.ReLU(max_value=6.0,
                          negative_slope=0.2)(x)

    if x.shape[-1] > 1 and dropout_amount > 0.0:
        x = k.layers.TimeDistributed(k.layers.SpatialDropout3D(dropout_amount,
                                                               trainable=train_bool),
                                     trainable=train_bool)(x)

    return x


def grouped_depthwise_conv3D_transpose(x, grouped_bool, grouped_channel_shuffle_bool, kernel_size, kernel_stride,
                                       kernel_dilation, ltwo_weight, lone_weight, linear_bool, gaussian_stddev,
                                       dropout_amount, train_bool):
    print("grouped_depthwise_conv3D_transpose")

    if grouped_bool and grouped_channel_shuffle_bool:
        x = k.layers.TimeDistributed(k.layers.Lambda(channel_shuffle,
                                                     arguments={"groups": x.shape[-1]}),
                                     trainable=train_bool)(x)

    x = k.layers.TimeDistributed(k.layers.Conv3DTranspose(filters=x.shape[-1],
                                                          kernel_size=kernel_size,
                                                          strides=kernel_stride,
                                                          dilation_rate=kernel_dilation,
                                                          groups=x.shape[-1],
                                                          padding="same",
                                                          kernel_initializer="he_uniform",
                                                          bias_initializer=k.initializers.Constant(0.0),
                                                          kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                          activity_regularizer=k.regularizers.l1(l1=lone_weight),
                                                          kernel_constraint=k.constraints.UnitNorm(),
                                                          trainable=train_bool),
                                 trainable=train_bool)(x)
    x = k.layers.BatchNormalization(trainable=train_bool)(x)

    if not linear_bool:
        if gaussian_stddev > 0.0:
            x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                       trainable=train_bool)(x)

        x = k.layers.ReLU(max_value=6.0,
                          negative_slope=0.2)(x)

    if x.shape[-1] > 1 and dropout_amount > 0.0:
        x = k.layers.TimeDistributed(k.layers.SpatialDropout3D(dropout_amount,
                                                               trainable=train_bool),
                                     trainable=train_bool)(x)

    return x


def get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, depth, size, stride, ltwo_weight,
                                    lone_weight, gaussian_stddev, dropout_amount, resnet_bool, train_bool,
                                    concatenate_bool, densenet_bool):
    print("get_transpose_convolution_layer")

    convolved_fov_bool = False
    dilated_bool = True
    depthwise_seperable_bool = parameters.depthwise_seperable_bool
    bottleneck_bool = parameters.bottleneck_bool

    convolved_fov_bool, kernel_padding, kernel_groups, fov_loops, kernel_size, kernel_dilation, kernel_depth = \
        conv3D_scaling(x, stride, dilated_bool, size, depth, grouped_bool, convolved_fov_bool, bottleneck_bool)

    res_connections = []

    for i in range(fov_loops):
        if i >= fov_loops - 1:
            kernel_stride = stride
        else:
            kernel_stride = (1, 1, 1)

        if resnet_bool:
            res_connections.append(x)

        if bottleneck_bool:
            x = bottleneck_conv3D_transpose(x, depth, grouped_bool, grouped_channel_shuffle_bool, kernel_groups,
                                            train_bool, ltwo_weight, lone_weight, False, gaussian_stddev,
                                            dropout_amount)

        if depthwise_seperable_bool:
            x = grouped_depthwise_conv3D_transpose(x, grouped_bool, grouped_channel_shuffle_bool, kernel_size,
                                                   kernel_stride, kernel_dilation, ltwo_weight, lone_weight, False,
                                                   gaussian_stddev, dropout_amount, train_bool)

            if grouped_bool and grouped_channel_shuffle_bool:
                x = k.layers.TimeDistributed(k.layers.Lambda(channel_shuffle,
                                                             arguments={"groups": kernel_groups}),
                                             trainable=train_bool)(x)

            x = k.layers.TimeDistributed(k.layers.Conv3DTranspose(filters=kernel_depth,
                                                                  kernel_size=(1, 1, 1),
                                                                  strides=(1, 1, 1),
                                                                  dilation_rate=(1, 1, 1),
                                                                  groups=kernel_groups,
                                                                  padding="same",
                                                                  kernel_initializer="he_uniform",
                                                                  bias_initializer=k.initializers.Constant(0.0),
                                                                  kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                                  activity_regularizer=
                                                                  k.regularizers.l1(l1=lone_weight),
                                                                  kernel_constraint=k.constraints.UnitNorm(),
                                                                  trainable=train_bool),
                                         trainable=train_bool)(x)
            x = k.layers.BatchNormalization(trainable=train_bool)(x)

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=train_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            if x.shape[-1] > 1 and dropout_amount > 0.0:
                x = k.layers.TimeDistributed(k.layers.SpatialDropout3D(dropout_amount,
                                                                       trainable=train_bool),
                                             trainable=train_bool)(x)

        else:
            if grouped_bool and grouped_channel_shuffle_bool:
                x = k.layers.TimeDistributed(k.layers.Lambda(channel_shuffle,
                                                             arguments={"groups": kernel_groups}),
                                             trainable=train_bool)(x)

            x = k.layers.TimeDistributed(k.layers.Conv3DTranspose(filters=kernel_depth,
                                                                  kernel_size=kernel_size,
                                                                  strides=kernel_stride,
                                                                  dilation_rate=kernel_dilation,
                                                                  groups=kernel_groups,
                                                                  padding="same",
                                                                  kernel_initializer="he_uniform",
                                                                  bias_initializer=k.initializers.Constant(0.0),
                                                                  kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                                  activity_regularizer=
                                                                  k.regularizers.l1(l1=lone_weight),
                                                                  kernel_constraint=k.constraints.UnitNorm(),
                                                                  trainable=train_bool),
                                         trainable=train_bool)(x)
            x = k.layers.BatchNormalization(trainable=train_bool)(x)

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=train_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            if x.shape[-1] > 1 and dropout_amount > 0.0:
                x = k.layers.TimeDistributed(k.layers.SpatialDropout3D(dropout_amount,
                                                                       trainable=train_bool),
                                             trainable=train_bool)(x)

        if bottleneck_bool:
            x = bottleneck_conv3D_transpose(x, depth, grouped_bool, grouped_channel_shuffle_bool, kernel_groups,
                                            train_bool, ltwo_weight, lone_weight, True, gaussian_stddev, dropout_amount)

        if resnet_bool and not (stride[0] > 1 or stride[1] > 1 or stride[2] > 1):
            if concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=train_bool)([x] + res_connections)

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=train_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                kernel_groups, train_bool, ltwo_weight, lone_weight, False,
                                                gaussian_stddev, dropout_amount)
            else:
                for j in range(len(res_connections)):
                    if res_connections[j].shape[-1] != x.shape[-1]:
                        kernel_groups = get_kernel_groups(res_connections[j], grouped_bool, x.shape[-1])

                        res_connections[j] = bottleneck_conv3D_transpose(res_connections[j], x.shape[-1], grouped_bool,
                                                                         grouped_channel_shuffle_bool,
                                                                         kernel_groups, train_bool, ltwo_weight,
                                                                         lone_weight, False, gaussian_stddev,
                                                                         dropout_amount)

                x = k.layers.Add(trainable=train_bool)([x] + res_connections)

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=train_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

            if not densenet_bool:
                res_connections = []
            else:
                res_connections = []

    return x
