# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow.keras as k


def get_convolution_layer(x, depth, size, stride, groups):
    print("get_convolution_layer")

    x = k.layers.Conv3D(filters=depth,
                        kernel_size=size,
                        strides=stride,
                        dilation_rate=(1, 1, 1),
                        groups=groups,
                        padding="same",
                        kernel_initializer="he_normal",
                        bias_initializer=k.initializers.Constant(0.0))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.ReLU(negative_slope=0.2)(x)

    return x
