# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow.keras as k


def get_convolution_layer(x, depth, size, stride):
    print("get_convolution_layer")

    kernel_depth = depth
    kernel_size = size
    kernel_stride = stride
    kernel_dilation = 1
    kernel_groups = 1

    x = k.layers.Conv3D(filters=kernel_depth,
                        kernel_size=kernel_size,
                        strides=kernel_stride,
                        dilation_rate=kernel_dilation,
                        groups=kernel_groups,
                        padding="same",
                        kernel_initializer="he_normal",
                        bias_initializer=k.initializers.Constant(0.0))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.ReLU(negative_slope=0.2)(x)

    return x


def get_transpose_convolution_layer(x, depth, size, stride):
    print("get_transpose_convolution_layer")

    kernel_depth = depth
    kernel_size = size
    kernel_stride = stride
    kernel_dilation = 1
    kernel_groups = 1

    x = k.layers.Conv3DTranspose(filters=kernel_depth,
                                 kernel_size=kernel_size,
                                 strides=kernel_stride,
                                 dilation_rate=kernel_dilation,
                                 groups=kernel_groups,
                                 padding="same",
                                 kernel_initializer="he_normal",
                                 bias_initializer=k.initializers.Constant(0.0))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.ReLU(negative_slope=0.2)(x)

    return x
