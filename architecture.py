# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import numpy as np
import tensorflow.keras as k

import parameters
import main
import layers
import loss


def get_input(input_shape):
    print("get_input")

    input_x = k.layers.Input(input_shape)

    return input_x, input_x


def get_encoder(x, grouped_bool, grouped_channel_shuffle_bool, input_gaussian_stddev, gaussian_stddev, ltwo_weight,
                lone_weight, dropout_amount):
    print("get_encoder")

    # settings extrapolated for layer 1 and 2
    layer_1_depth = 8
    layer_2_depth = 8

    layer_3_depth = 8
    layer_4_depth = 4
    layer_5_depth = 4
    layer_6_depth = 2
    to_latent_depth = 2

    # settings extrapolated for layer 1 and 2
    layer_1_kernel_size = 3
    layer_2_kernel_size = 3

    layer_3_kernel_size = 3
    layer_4_kernel_size = 3
    layer_5_kernel_size = 3
    layer_6_kernel_size = 3
    to_latent_kernel_size = 3

    # settings extrapolated for layer 1 and 2
    layer_1_layers = 2
    layer_2_layers = 2

    layer_3_layers = 2
    layer_4_layers = 2
    layer_5_layers = 2
    layer_6_layers = 2
    to_latent_layers = 1

    if input_gaussian_stddev > 0.0:
        x = k.layers.GaussianNoise(stddev=input_gaussian_stddev,
                                   trainable=main.autoencoder_bool or main.classifier_bool)(x)

    res_connections = []

    # layer 1
    if main.cluster_bool:
        for i in range(layer_1_layers):
            x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_1_depth, (3, 3, 3),
                                             (layer_1_kernel_size, layer_1_kernel_size, layer_1_kernel_size),
                                             ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                             main.autoencoder_resnet_bool,
                                             main.autoencoder_bool or main.classifier_bool,
                                             main.autoencoder_resnet_concatenate_bool, main.autoencoder_densenet_bool)

        if gaussian_stddev > 0.0:
            res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                          trainable=main.autoencoder_bool or main.classifier_bool)(x))
        else:
            res_connections.append(x)

        if main.down_stride_bool:
            if main.down_max_pool_too_bool:
                x_1 = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_1_depth,
                                                   (layer_1_kernel_size, layer_1_kernel_size, layer_1_kernel_size),
                                                   (2, 2, 2), ltwo_weight, lone_weight, gaussian_stddev,
                                                   dropout_amount, main.autoencoder_resnet_bool,
                                                   main.autoencoder_bool or main.classifier_bool,
                                                   main.autoencoder_resnet_concatenate_bool,
                                                   main.autoencoder_densenet_bool)

                x_2 = k.layers.TimeDistributed(k.layers.AveragePooling3D(pool_size=(3, 3, 3),
                                                                         strides=(2, 2, 2),
                                                                         padding="same",
                                                                         trainable=main.autoencoder_bool or
                                                                                   main.classifier_bool),
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                if main.down_max_pool_too_concatenate_bool:
                    concatenate_depth = x.shape[-1]

                    x = k.layers.Concatenate(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                   trainable=main.autoencoder_bool or main.classifier_bool)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)

                    kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                    x = layers.bottleneck_conv3D(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                 kernel_groups, main.autoencoder_bool or main.classifier_bool,
                                                 ltwo_weight, lone_weight, False, gaussian_stddev, dropout_amount)
                else:
                    x = k.layers.Add(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                   trainable=main.autoencoder_bool or main.classifier_bool)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)
            else:
                x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_1_depth,
                                                 (layer_1_kernel_size, layer_1_kernel_size, layer_1_kernel_size),
                                                 (2, 2, 2), ltwo_weight, lone_weight, gaussian_stddev,
                                                 dropout_amount, main.autoencoder_resnet_bool,
                                                 main.autoencoder_bool or main.classifier_bool,
                                                 main.autoencoder_resnet_concatenate_bool,
                                                 main.autoencoder_densenet_bool)
        else:
            x = k.layers.TimeDistributed(k.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                                               padding="same",
                                                               trainable=main.autoencoder_bool or main.classifier_bool),
                                         trainable=main.autoencoder_bool or main.classifier_bool)(x)

    # layer 2
    if main.cluster_bool:
        for i in range(layer_2_layers):
            x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_2_depth,
                                             (layer_2_kernel_size, layer_2_kernel_size, layer_2_kernel_size),
                                             (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                             main.autoencoder_resnet_bool,
                                             main.autoencoder_bool or main.classifier_bool,
                                             main.autoencoder_resnet_concatenate_bool, main.autoencoder_densenet_bool)

        if gaussian_stddev > 0.0:
            res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                          trainable=main.autoencoder_bool or main.classifier_bool)(x))
        else:
            res_connections.append(x)

        if main.down_stride_bool:
            if main.down_max_pool_too_bool:
                x_1 = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_2_depth,
                                                   (layer_2_kernel_size, layer_2_kernel_size, layer_2_kernel_size),
                                                   (2, 2, 2), ltwo_weight, lone_weight, gaussian_stddev,
                                                   dropout_amount, main.autoencoder_resnet_bool,
                                                   main.autoencoder_bool or main.classifier_bool,
                                                   main.autoencoder_resnet_concatenate_bool,
                                                   main.autoencoder_densenet_bool)

                x_2 = k.layers.TimeDistributed(k.layers.AveragePooling3D(pool_size=(3, 3, 3),
                                                                         strides=(2, 2, 2),
                                                                         padding="same",
                                                                         trainable=main.autoencoder_bool or
                                                                                   main.classifier_bool),
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                if main.down_max_pool_too_concatenate_bool:
                    concatenate_depth = x.shape[-1]

                    x = k.layers.Concatenate(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                   trainable=main.autoencoder_bool or main.classifier_bool)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)

                    kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                    x = layers.bottleneck_conv3D(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                 kernel_groups, main.autoencoder_bool or main.classifier_bool,
                                                 ltwo_weight, lone_weight, False, gaussian_stddev, dropout_amount)
                else:
                    x = k.layers.Add(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                   trainable=main.autoencoder_bool or main.classifier_bool)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)
            else:
                x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_2_depth,
                                                 (layer_2_kernel_size, layer_2_kernel_size, layer_2_kernel_size),
                                                 (2, 2, 2), ltwo_weight, lone_weight, gaussian_stddev,
                                                 dropout_amount, main.autoencoder_resnet_bool,
                                                 main.autoencoder_bool or main.classifier_bool,
                                                 main.autoencoder_resnet_concatenate_bool,
                                                 main.autoencoder_densenet_bool)
        else:
            x = k.layers.TimeDistributed(k.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                                               padding="same",
                                                               trainable=main.autoencoder_bool or main.classifier_bool),
                                         trainable=main.autoencoder_bool or main.classifier_bool)(x)

    # layer 3
    for i in range(layer_3_layers):
        x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_3_depth,
                                         (layer_3_kernel_size, layer_3_kernel_size, layer_3_kernel_size), (1, 1, 1),
                                         ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                         main.autoencoder_resnet_bool, main.autoencoder_bool or main.classifier_bool,
                                         main.autoencoder_resnet_concatenate_bool, main.autoencoder_densenet_bool)

    if gaussian_stddev > 0.0:
        res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                      trainable=main.autoencoder_bool or main.classifier_bool)(x))
    else:
        res_connections.append(x)

    if main.down_stride_bool:
        if main.down_max_pool_too_bool:
            x_1 = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_3_depth,
                                               (layer_3_kernel_size, layer_3_kernel_size, layer_3_kernel_size),
                                               (2, 2, 2), ltwo_weight, lone_weight, gaussian_stddev,
                                               dropout_amount, main.autoencoder_resnet_bool,
                                               main.autoencoder_bool or main.classifier_bool,
                                               main.autoencoder_resnet_concatenate_bool,
                                               main.autoencoder_densenet_bool)

            x_2 = k.layers.TimeDistributed(k.layers.AveragePooling3D(pool_size=(3, 3, 3),
                                                                     strides=(2, 2, 2),
                                                                     padding="same",
                                                                     trainable=main.autoencoder_bool or
                                                                               main.classifier_bool),
                                           trainable=main.autoencoder_bool or main.classifier_bool)(x)

            if main.down_max_pool_too_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                             kernel_groups, main.autoencoder_bool or main.classifier_bool, ltwo_weight,
                                             lone_weight, False, gaussian_stddev, dropout_amount)
            else:
                x = k.layers.Add(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)
        else:
            x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_3_depth,
                                             (layer_3_kernel_size, layer_3_kernel_size, layer_3_kernel_size), (2, 2, 2),
                                             ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                             main.autoencoder_resnet_bool,
                                             main.autoencoder_bool or main.classifier_bool,
                                             main.autoencoder_resnet_concatenate_bool,
                                             main.autoencoder_densenet_bool)
    else:
        x = k.layers.TimeDistributed(k.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                                           padding="same",
                                                           trainable=main.autoencoder_bool or main.classifier_bool),
                                     trainable=main.autoencoder_bool or main.classifier_bool)(x)

    # layer 4
    for i in range(layer_4_layers):
        x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_4_depth,
                                         (layer_4_kernel_size, layer_4_kernel_size, layer_4_kernel_size), (1, 1, 1),
                                         ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                         main.autoencoder_resnet_bool, main.autoencoder_bool or main.classifier_bool,
                                         main.autoencoder_resnet_concatenate_bool, main.autoencoder_densenet_bool)

    if gaussian_stddev > 0.0:
        res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                      trainable=main.autoencoder_bool or main.classifier_bool)(x))
    else:
        res_connections.append(x)

    if main.down_stride_bool:
        if main.down_max_pool_too_bool:
            x_1 = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_4_depth,
                                               (layer_4_kernel_size, layer_4_kernel_size, layer_4_kernel_size),
                                               (2, 2, 2), ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                               main.autoencoder_resnet_bool,
                                               main.autoencoder_bool or main.classifier_bool,
                                               main.autoencoder_resnet_concatenate_bool,
                                               main.autoencoder_densenet_bool)

            x_2 = k.layers.TimeDistributed(k.layers.AveragePooling3D(pool_size=(3, 3, 3),
                                                                     strides=(2, 2, 2),
                                                                     padding="same",
                                                                     trainable=main.autoencoder_bool or
                                                                               main.classifier_bool),
                                           trainable=main.autoencoder_bool or main.classifier_bool)(x)

            if main.down_max_pool_too_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                             kernel_groups, main.autoencoder_bool or main.classifier_bool, ltwo_weight,
                                             lone_weight, False, gaussian_stddev, dropout_amount)
            else:
                x = k.layers.Add(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)
        else:
            x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_4_depth,
                                             (layer_4_kernel_size, layer_4_kernel_size, layer_4_kernel_size), (2, 2, 2),
                                             ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                             main.autoencoder_resnet_bool,
                                             main.autoencoder_bool or main.classifier_bool,
                                             main.autoencoder_resnet_concatenate_bool,
                                             main.autoencoder_densenet_bool)
    else:
        x = k.layers.TimeDistributed(k.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                                           padding="same",
                                                           trainable=main.autoencoder_bool or main.classifier_bool),
                                     trainable=main.autoencoder_bool or main.classifier_bool)(x)

    # layer 5
    for i in range(layer_5_layers):
        x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_5_depth,
                                         (layer_5_kernel_size, layer_5_kernel_size, layer_5_kernel_size), (1, 1, 1),
                                         ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                         main.autoencoder_resnet_bool, main.autoencoder_bool or main.classifier_bool,
                                         main.autoencoder_resnet_concatenate_bool, main.autoencoder_densenet_bool)

    if gaussian_stddev > 0.0:
        res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                      trainable=main.autoencoder_bool or main.classifier_bool)(x))
    else:
        res_connections.append(x)

    if main.down_stride_bool:
        if main.down_max_pool_too_bool:
            x_1 = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_5_depth,
                                               (layer_5_kernel_size, layer_5_kernel_size, layer_5_kernel_size),
                                               (2, 2, 1), ltwo_weight, lone_weight, gaussian_stddev,
                                               dropout_amount, main.autoencoder_resnet_bool,
                                               main.autoencoder_bool or main.classifier_bool,
                                               main.autoencoder_resnet_concatenate_bool,
                                               main.autoencoder_densenet_bool)

            x_2 = k.layers.TimeDistributed(k.layers.AveragePooling3D(pool_size=(3, 3, 3),
                                                                     strides=(2, 2, 1),
                                                                     padding="same",
                                                                     trainable=main.autoencoder_bool or
                                                                               main.classifier_bool),
                                           trainable=main.autoencoder_bool or main.classifier_bool)(x)
            if main.down_max_pool_too_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                             kernel_groups, main.autoencoder_bool or main.classifier_bool, ltwo_weight,
                                             lone_weight, False, gaussian_stddev, dropout_amount)
            else:
                x = k.layers.Add(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)
        else:
            x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_5_depth,
                                             (layer_5_kernel_size, layer_5_kernel_size, layer_5_kernel_size), (2, 2, 1),
                                             ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                             main.autoencoder_resnet_bool,
                                             main.autoencoder_bool or main.classifier_bool,
                                             main.autoencoder_resnet_concatenate_bool,
                                             main.autoencoder_densenet_bool)
    else:
        x = k.layers.TimeDistributed(k.layers.MaxPooling3D(pool_size=(2, 2, 1),
                                                           padding="same",
                                                           trainable=main.autoencoder_bool or main.classifier_bool),
                                     trainable=main.autoencoder_bool or main.classifier_bool)(x)

    # layer 6
    for i in range(layer_6_layers):
        x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_6_depth,
                                         (layer_6_kernel_size, layer_6_kernel_size, layer_6_kernel_size), (1, 1, 1),
                                         ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                         main.autoencoder_resnet_bool, main.autoencoder_bool or main.classifier_bool,
                                         main.autoencoder_resnet_concatenate_bool, main.autoencoder_densenet_bool)

    if gaussian_stddev > 0.0:
        res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                      trainable=main.autoencoder_bool or main.classifier_bool)(x))
    else:
        res_connections.append(x)

    if main.down_stride_bool:
        if main.down_max_pool_too_bool:
            x_1 = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_6_depth,
                                               (layer_6_kernel_size, layer_6_kernel_size, layer_6_kernel_size),
                                               (1, 2, 1), ltwo_weight, lone_weight, gaussian_stddev,
                                               dropout_amount, main.autoencoder_resnet_bool,
                                               main.autoencoder_bool or main.classifier_bool,
                                               main.autoencoder_resnet_concatenate_bool,
                                               main.autoencoder_densenet_bool)

            x_2 = k.layers.TimeDistributed(k.layers.AveragePooling3D(pool_size=(3, 3, 3),
                                                                     strides=(1, 2, 1),
                                                                     padding="same",
                                                                     trainable=main.autoencoder_bool or
                                                                               main.classifier_bool),
                                           trainable=main.autoencoder_bool or main.classifier_bool)(x)
            if main.down_max_pool_too_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                             kernel_groups, main.autoencoder_bool or main.classifier_bool, ltwo_weight,
                                             lone_weight, False, gaussian_stddev, dropout_amount)

                if x.shape[-1] > 1 and dropout_amount > 0.0:
                    x = k.layers.TimeDistributed(k.layers.SpatialDropout3D(dropout_amount,
                                                                           trainable=main.autoencoder_bool or
                                                                                     main.classifier_bool),
                                                 trainable=main.autoencoder_bool or main.classifier_bool)(x)
            else:
                x = k.layers.Add(trainable=main.autoencoder_bool or main.classifier_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool or main.classifier_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)
        else:
            x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_6_depth,
                                             (layer_6_kernel_size, layer_6_kernel_size, layer_6_kernel_size), (1, 2, 1),
                                             ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                             main.autoencoder_resnet_bool,
                                             main.autoencoder_bool or main.classifier_bool,
                                             main.autoencoder_resnet_concatenate_bool,
                                             main.autoencoder_densenet_bool)
    else:
        x = k.layers.TimeDistributed(k.layers.MaxPooling3D(pool_size=(1, 2, 1),
                                                           padding="same",
                                                           trainable=main.autoencoder_bool or main.classifier_bool),
                                     trainable=main.autoencoder_bool or main.classifier_bool)(x)

    # to latent
    for i in range(to_latent_layers):
        x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, to_latent_depth,
                                         (to_latent_kernel_size, to_latent_kernel_size, to_latent_kernel_size),
                                         (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                         main.autoencoder_resnet_bool, main.autoencoder_bool or main.classifier_bool,
                                         main.autoencoder_resnet_concatenate_bool, main.autoencoder_densenet_bool)

    if gaussian_stddev > 0.0:
        res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                      trainable=main.autoencoder_bool or main.classifier_bool)(x))
    else:
        res_connections.append(x)

    return x, res_connections


def get_latent(x, grouped_bool, latent_size, ltwo_weight, lone_weight, gaussian_stddev, dropout_amount):
    print("get_latent")

    mu, sigma = None, None

    if main.cnn_only_autoencoder_bool:
        latent_stride = (1, 1, 1)
        latent_kernel_size = (1, 1, 1)

        convolved_fov_bool, kernel_padding, kernel_groups, fov_loops, kernel_size, latent_kernel_dilation, \
        kernel_depth = layers.conv3D_scaling(x, latent_stride, False, latent_kernel_size, latent_size, grouped_bool,
                                             False, False)

        x = k.layers.TimeDistributed(k.layers.Conv3D(filters=x.shape[-1],
                                                     kernel_size=latent_kernel_size,
                                                     strides=latent_stride,
                                                     dilation_rate=latent_kernel_dilation,
                                                     groups=kernel_groups,
                                                     padding="valid",
                                                     kernel_initializer="glorot_normal",
                                                     bias_initializer=k.initializers.Constant(0.0),
                                                     trainable=main.autoencoder_bool),
                                     trainable=main.autoencoder_bool,
                                     name="latent")(x)

        x_shape = x.shape

        x = k.layers.TimeDistributed(k.layers.Flatten(trainable=main.autoencoder_bool),
                                     trainable=main.autoencoder_bool)(x)

        x_latent_center = x

        x = k.layers.TimeDistributed(k.layers.Reshape((x_shape[2], x_shape[3], x_shape[4], x_shape[5])),
                                     trainable=main.autoencoder_bool)(x)
    else:
        x_shape = x.shape

        x = k.layers.TimeDistributed(k.layers.Flatten(trainable=main.autoencoder_bool),
                                     trainable=main.autoencoder_bool)(x)

        if main.dv_bool:
            x, mu, sigma = layers.get_dv_latent_layer(x, latent_size, main.autoencoder_bool)
        else:
            x = k.layers.Bidirectional(k.layers.GRU(units=latent_size,
                                                    activation="tanh",
                                                    recurrent_activation="sigmoid",
                                                    use_bias=True,
                                                    kernel_initializer="glorot_uniform",
                                                    recurrent_initializer="glorot_uniform",
                                                    bias_initializer=k.initializers.Constant(0.0),
                                                    kernel_regularizer=None,
                                                    activity_regularizer=None,
                                                    kernel_constraint=None,
                                                    dropout=0.0,
                                                    return_sequences=True,
                                                    trainable=main.autoencoder_bool),
                                       merge_mode="sum",
                                       trainable=main.autoencoder_bool,
                                       name="latent")(x)

        x_latent_center = x

        layer_size = np.prod(x_shape[2:])

        x = k.layers.TimeDistributed(k.layers.Dense(units=layer_size,
                                                    kernel_initializer="glorot_normal",
                                                    bias_initializer=k.initializers.Constant(0.0),
                                                    kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                    activity_regularizer=k.regularizers.l1(l1=lone_weight),
                                                    kernel_constraint=k.constraints.UnitNorm(),
                                                    trainable=main.autoencoder_bool),
                                     trainable=main.autoencoder_bool)(x)
        x = k.layers.LayerNormalization(trainable=main.autoencoder_bool)(x)

        if gaussian_stddev > 0.0:
            x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                       trainable=main.autoencoder_bool)(x)

        x = k.layers.Activation("tanh",
                                trainable=main.autoencoder_bool)(x)

        if dropout_amount > 0.0:
            x = k.layers.TimeDistributed(k.layers.Dropout(dropout_amount,
                                                          trainable=main.autoencoder_bool),
                                         trainable=main.autoencoder_bool)(x)

        x = k.layers.TimeDistributed(k.layers.Reshape((x_shape[2], x_shape[3], x_shape[4], x_shape[5])))(x)

    return x, x_latent_center, mu, sigma


def get_classifier(x, output_layer_size, ltwo_weight, lone_weight, dropout_amount, gaussian_stddev):
    print("get_classifier")

    instantiate_classifier_bool = True

    if instantiate_classifier_bool:
        res_connections = []

        if main.classifier_resnet_bool:
            if gaussian_stddev > 0.0:
                res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                              trainable=main.classifier_bool)(x))
            else:
                res_connections.append(x)

        layer_size = (output_layer_size * 2) * 2

        x = k.layers.Bidirectional(k.layers.GRU(units=layer_size,
                                                activation="tanh",
                                                recurrent_activation="sigmoid",
                                                use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="glorot_uniform",
                                                bias_initializer=k.initializers.Constant(0.0),
                                                kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                activity_regularizer=k.regularizers.l1(l1=lone_weight),
                                                kernel_constraint=k.constraints.UnitNorm(),
                                                dropout=dropout_amount,
                                                return_sequences=True,
                                                trainable=main.classifier_bool),
                                   merge_mode="sum",
                                   trainable=main.classifier_bool)(x)

        if main.classifier_resnet_bool:
            x = k.layers.Concatenate(trainable=main.classifier_bool)([x] + res_connections)

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.classifier_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            if not main.classifier_densenet_bool:
                res_connections = []

            if gaussian_stddev > 0.0:
                res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                              trainable=main.classifier_bool)(x))
            else:
                res_connections.append(x)

        layer_size = output_layer_size * 2

        x = k.layers.Bidirectional(k.layers.GRU(units=layer_size,
                                                activation="tanh",
                                                recurrent_activation="sigmoid",
                                                use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="glorot_uniform",
                                                bias_initializer=k.initializers.Constant(0.0),
                                                kernel_regularizer=k.regularizers.l2(l2=ltwo_weight),
                                                activity_regularizer=k.regularizers.l1(l1=lone_weight),
                                                kernel_constraint=k.constraints.UnitNorm(),
                                                dropout=dropout_amount,
                                                return_sequences=True,
                                                trainable=main.classifier_bool),
                                   merge_mode="sum",
                                   trainable=main.classifier_bool)(x)

        if main.classifier_resnet_bool:
            x = k.layers.Concatenate(trainable=main.classifier_bool)([x] + res_connections)

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.classifier_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

        x = k.layers.Bidirectional(k.layers.GRU(units=output_layer_size,
                                                activation="tanh",
                                                recurrent_activation="sigmoid",
                                                use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="glorot_uniform",
                                                bias_initializer=k.initializers.Constant(0.0),
                                                kernel_regularizer=None,
                                                activity_regularizer=None,
                                                kernel_constraint=None,
                                                dropout=0.0,
                                                return_sequences=True,
                                                trainable=main.classifier_bool),
                                   merge_mode="sum",
                                   trainable=main.classifier_bool)(x)

    x = k.layers.TimeDistributed(k.layers.Dense(units=output_layer_size,
                                                kernel_initializer="glorot_normal",
                                                bias_initializer=k.initializers.Constant(0.0),
                                                trainable=main.classifier_bool),
                                 trainable=main.classifier_bool,
                                 name="output_1")(x)

    return x


def get_decoder(x, grouped_bool, grouped_channel_shuffle_bool, res_connections, ltwo_weight, lone_weight,
                gaussian_stddev, dropout_amount):
    print("get_decoder")

    from_latent_depth = 2
    layer_6_depth = 2
    layer_5_depth = 4
    layer_4_depth = 4
    layer_3_depth = 8

    # settings extrapolated for layer 1 and 2
    layer_2_depth = 8
    layer_1_depth = 8

    from_latent_kernel_size = 3
    layer_6_kernel_size = 3
    layer_5_kernel_size = 3
    layer_4_kernel_size = 3
    layer_3_kernel_size = 3

    # settings extrapolated for layer 1 and 2
    layer_2_kernel_size = 3
    layer_1_kernel_size = 3

    from_latent_layers = 1
    layer_6_layers = 2
    layer_5_layers = 2
    layer_4_layers = 2
    layer_3_layers = 2

    # settings extrapolated for layer 1 and 2
    layer_2_layers = 2
    layer_1_layers = 2

    # from latent
    if main.autoencoder_unet_bool:
        res_connections_x = res_connections.pop()

        if main.autoencoder_unet_concatenate_bool:
            concatenate_depth = x.shape[-1]

            x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

            x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                   kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                   False, gaussian_stddev, dropout_amount)
        else:
            x = k.layers.Add(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

    for i in range(from_latent_layers):
        x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, from_latent_depth,
                                                   (from_latent_kernel_size,
                                                    from_latent_kernel_size,
                                                    from_latent_kernel_size), (1, 1, 1), ltwo_weight, lone_weight,
                                                   gaussian_stddev, dropout_amount, main.autoencoder_resnet_bool,
                                                   main.autoencoder_bool,
                                                   main.autoencoder_resnet_concatenate_bool,
                                                   main.autoencoder_densenet_bool)

    # layer 6
    if main.up_stride_bool:
        if main.up_upsample_too_bool:
            x_1 = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                         layer_6_depth, (layer_6_kernel_size,
                                                                         layer_6_kernel_size,
                                                                         layer_6_kernel_size), (1, 2, 1), ltwo_weight,
                                                         lone_weight, gaussian_stddev, dropout_amount,
                                                         main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                         main.autoencoder_resnet_concatenate_bool,
                                                         main.autoencoder_densenet_bool)

            x_2 = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(1, 2, 1),
                                                                 trainable=main.autoencoder_bool),
                                           trainable=main.autoencoder_bool)(x)

            if main.down_max_pool_too_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                       kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                       False, gaussian_stddev, dropout_amount)
            else:
                if x_2.shape != x_1.shape:
                    if gaussian_stddev > 0.0:
                        x_2 = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                     trainable=main.autoencoder_bool)(x_2)

                    x_2 = k.layers.ReLU(max_value=6.0,
                                        negative_slope=0.2)(x_2)

                    kernel_groups = layers.get_kernel_groups(x_2, grouped_bool, x_1.shape[-1])

                    x_2 = layers.bottleneck_conv3D_transpose(x_2, x_1.shape[-1], grouped_bool,
                                                             grouped_channel_shuffle_bool,
                                                             kernel_groups, main.autoencoder_bool, ltwo_weight,
                                                             lone_weight,
                                                             False, gaussian_stddev, dropout_amount)

                x = k.layers.Add(trainable=main.autoencoder_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)
        else:
            x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                       layer_6_depth, (layer_6_kernel_size,
                                                                       layer_6_kernel_size,
                                                                       layer_6_kernel_size), (1, 2, 1), ltwo_weight,
                                                       lone_weight, gaussian_stddev, dropout_amount,
                                                       main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                       main.autoencoder_resnet_concatenate_bool,
                                                       main.autoencoder_densenet_bool)
    else:
        x = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(1, 2, 1),
                                                           trainable=main.autoencoder_bool),
                                     trainable=main.autoencoder_bool)(x)

    if main.autoencoder_unet_bool:
        res_connections_x = res_connections.pop()

        if main.autoencoder_unet_concatenate_bool:
            concatenate_depth = x.shape[-1]

            x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

            x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                   kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                   False, gaussian_stddev, dropout_amount)
        else:
            x = k.layers.Add(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

    for i in range(layer_6_layers):
        x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_6_depth,
                                                   (layer_6_kernel_size, layer_6_kernel_size, layer_6_kernel_size),
                                                   (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                                   main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                   main.autoencoder_resnet_concatenate_bool,
                                                   main.autoencoder_densenet_bool)

    # layer 5
    if main.up_stride_bool:
        if main.up_upsample_too_bool:
            x_1 = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                         layer_5_depth, (layer_5_kernel_size,
                                                                         layer_5_kernel_size,
                                                                         layer_5_kernel_size), (2, 2, 1), ltwo_weight,
                                                         lone_weight, gaussian_stddev, dropout_amount,
                                                         main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                         main.autoencoder_resnet_concatenate_bool,
                                                         main.autoencoder_densenet_bool)

            x_2 = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 1),
                                                                 trainable=main.autoencoder_bool),
                                           trainable=main.autoencoder_bool)(x)

            if main.down_max_pool_too_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                       kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                       False, gaussian_stddev, dropout_amount)
            else:
                if x_2.shape != x_1.shape:
                    if gaussian_stddev > 0.0:
                        x_2 = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                     trainable=main.autoencoder_bool)(x_2)

                    x_2 = k.layers.ReLU(max_value=6.0,
                                        negative_slope=0.2)(x_2)

                    kernel_groups = layers.get_kernel_groups(x_2, grouped_bool, x_1.shape[-1])

                    x_2 = layers.bottleneck_conv3D_transpose(x_2, x_1.shape[-1], grouped_bool,
                                                             grouped_channel_shuffle_bool,
                                                             kernel_groups, main.autoencoder_bool, ltwo_weight,
                                                             lone_weight,
                                                             False, gaussian_stddev, dropout_amount)

                x = k.layers.Add(trainable=main.autoencoder_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)
        else:
            x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                       layer_5_depth, (layer_5_kernel_size,
                                                                       layer_5_kernel_size,
                                                                       layer_5_kernel_size), (2, 2, 1), ltwo_weight,
                                                       lone_weight, gaussian_stddev, dropout_amount,
                                                       main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                       main.autoencoder_resnet_concatenate_bool,
                                                       main.autoencoder_densenet_bool)
    else:
        x = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 1),
                                                           trainable=main.autoencoder_bool),
                                     trainable=main.autoencoder_bool)(x)

    if main.autoencoder_unet_bool:
        res_connections_x = res_connections.pop()

        if main.autoencoder_unet_concatenate_bool:
            concatenate_depth = x.shape[-1]

            x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

            x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                   kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                   False, gaussian_stddev, dropout_amount)
        else:
            x = k.layers.Add(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

    for i in range(layer_5_layers):
        x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_5_depth,
                                                   (layer_5_kernel_size, layer_5_kernel_size, layer_5_kernel_size),
                                                   (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                                   main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                   main.autoencoder_resnet_concatenate_bool,
                                                   main.autoencoder_densenet_bool)

    # layer 4
    if main.up_stride_bool:
        if main.up_upsample_too_bool:
            x_1 = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                         layer_4_depth, (layer_4_kernel_size,
                                                                         layer_4_kernel_size,
                                                                         layer_4_kernel_size), (2, 2, 2), ltwo_weight,
                                                         lone_weight, gaussian_stddev, dropout_amount,
                                                         main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                         main.autoencoder_resnet_concatenate_bool,
                                                         main.autoencoder_densenet_bool)

            x_2 = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 2),
                                                                 trainable=main.autoencoder_bool),
                                           trainable=main.autoencoder_bool)(x)

            if main.down_max_pool_too_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                       kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                       False, gaussian_stddev, dropout_amount)
            else:
                if x_2.shape != x_1.shape:
                    if gaussian_stddev > 0.0:
                        x_2 = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                     trainable=main.autoencoder_bool)(x_2)

                    x_2 = k.layers.ReLU(max_value=6.0,
                                        negative_slope=0.2)(x_2)

                    kernel_groups = layers.get_kernel_groups(x_2, grouped_bool, x_1.shape[-1])

                    x_2 = layers.bottleneck_conv3D_transpose(x_2, x_1.shape[-1], grouped_bool,
                                                             grouped_channel_shuffle_bool,
                                                             kernel_groups, main.autoencoder_bool, ltwo_weight,
                                                             lone_weight,
                                                             False, gaussian_stddev, dropout_amount)

                x = k.layers.Add(trainable=main.autoencoder_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)
        else:
            x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                       layer_4_depth, (layer_4_kernel_size,
                                                                       layer_4_kernel_size,
                                                                       layer_4_kernel_size), (2, 2, 2), ltwo_weight,
                                                       lone_weight, gaussian_stddev, dropout_amount,
                                                       main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                       main.autoencoder_resnet_concatenate_bool,
                                                       main.autoencoder_densenet_bool)
    else:
        x = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 2),
                                                           trainable=main.autoencoder_bool),
                                     trainable=main.autoencoder_bool)(x)

    if main.autoencoder_unet_bool:
        res_connections_x = res_connections.pop()

        if main.autoencoder_unet_concatenate_bool:
            concatenate_depth = x.shape[-1]

            x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

            x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                   kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                   False, gaussian_stddev, dropout_amount)
        else:
            x = k.layers.Add(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

    for i in range(layer_4_layers):
        x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_4_depth,
                                                   (layer_4_kernel_size, layer_4_kernel_size, layer_4_kernel_size),
                                                   (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                                   main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                   main.autoencoder_resnet_concatenate_bool,
                                                   main.autoencoder_densenet_bool)

    # layer 3
    if main.up_stride_bool:
        if main.up_upsample_too_bool:
            x_1 = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                         layer_3_depth, (layer_3_kernel_size,
                                                                         layer_3_kernel_size,
                                                                         layer_3_kernel_size), (2, 2, 2), ltwo_weight,
                                                         lone_weight, gaussian_stddev, dropout_amount,
                                                         main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                         main.autoencoder_resnet_concatenate_bool,
                                                         main.autoencoder_densenet_bool)

            x_2 = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 2),
                                                                 trainable=main.autoencoder_bool),
                                           trainable=main.autoencoder_bool)(x)

            if main.down_max_pool_too_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                       kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                       False, gaussian_stddev, dropout_amount)
            else:
                if x_2.shape != x_1.shape:
                    if gaussian_stddev > 0.0:
                        x_2 = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                     trainable=main.autoencoder_bool)(x_2)

                    x_2 = k.layers.ReLU(max_value=6.0,
                                        negative_slope=0.2)(x_2)

                    kernel_groups = layers.get_kernel_groups(x_2, grouped_bool, x_1.shape[-1])

                    x_2 = layers.bottleneck_conv3D_transpose(x_2, x_1.shape[-1], grouped_bool,
                                                             grouped_channel_shuffle_bool,
                                                             kernel_groups, main.autoencoder_bool, ltwo_weight,
                                                             lone_weight,
                                                             False, gaussian_stddev, dropout_amount)

                x = k.layers.Add(trainable=main.autoencoder_bool)([x_1, x_2])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)
        else:
            x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                       layer_3_depth, (layer_3_kernel_size,
                                                                       layer_3_kernel_size,
                                                                       layer_3_kernel_size), (2, 2, 2), ltwo_weight,
                                                       lone_weight, gaussian_stddev, dropout_amount,
                                                       main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                       main.autoencoder_resnet_concatenate_bool,
                                                       main.autoencoder_densenet_bool)
    else:
        x = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 2),
                                                           trainable=main.autoencoder_bool),
                                     trainable=main.autoencoder_bool)(x)

    if main.autoencoder_unet_bool:
        res_connections_x = res_connections.pop()

        if main.autoencoder_unet_concatenate_bool:
            concatenate_depth = x.shape[-1]

            x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

            kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

            x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                   kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                   False, gaussian_stddev, dropout_amount)
        else:
            x = k.layers.Add(trainable=main.autoencoder_bool)([x, res_connections_x])

            if gaussian_stddev > 0.0:
                x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                           trainable=main.autoencoder_bool)(x)

            x = k.layers.ReLU(max_value=6.0,
                              negative_slope=0.2)(x)

    for i in range(layer_3_layers):
        x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_3_depth,
                                                   (layer_3_kernel_size, layer_3_kernel_size, layer_3_kernel_size),
                                                   (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                                   main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                   main.autoencoder_resnet_concatenate_bool,
                                                   main.autoencoder_densenet_bool)

    # layer 2
    if main.cluster_bool:
        if main.up_stride_bool:
            if main.up_upsample_too_bool:
                x_1 = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                             layer_2_depth, (layer_2_kernel_size,
                                                                             layer_2_kernel_size,
                                                                             layer_2_kernel_size), (2, 2, 2),
                                                             ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                                             main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                             main.autoencoder_resnet_concatenate_bool,
                                                             main.autoencoder_densenet_bool)

                x_2 = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 2),
                                                                     trainable=main.autoencoder_bool),
                                               trainable=main.autoencoder_bool)(x)

                if main.down_max_pool_too_concatenate_bool:
                    concatenate_depth = x.shape[-1]

                    x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                   trainable=main.autoencoder_bool)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)

                    kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                    x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool,
                                                           grouped_channel_shuffle_bool, kernel_groups,
                                                           main.autoencoder_bool, ltwo_weight, lone_weight,
                                                           False, gaussian_stddev, dropout_amount)
                else:
                    if x_2.shape != x_1.shape:
                        if gaussian_stddev > 0.0:
                            x_2 = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                         trainable=main.autoencoder_bool)(x_2)

                        x_2 = k.layers.ReLU(max_value=6.0,
                                            negative_slope=0.2)(x_2)

                        kernel_groups = layers.get_kernel_groups(x_2, grouped_bool, x_1.shape[-1])

                        x_2 = layers.bottleneck_conv3D_transpose(x_2, x_1.shape[-1], grouped_bool,
                                                                 grouped_channel_shuffle_bool,
                                                                 kernel_groups, main.autoencoder_bool, ltwo_weight,
                                                                 lone_weight,
                                                                 False, gaussian_stddev, dropout_amount)

                    x = k.layers.Add(trainable=main.autoencoder_bool)([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                   trainable=main.autoencoder_bool)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)
            else:
                x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                           layer_2_depth, (layer_2_kernel_size,
                                                                           layer_2_kernel_size,
                                                                           layer_2_kernel_size), (2, 2, 2), ltwo_weight,
                                                           lone_weight, gaussian_stddev, dropout_amount,
                                                           main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                           main.autoencoder_resnet_concatenate_bool,
                                                           main.autoencoder_densenet_bool)
        else:
            x = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 2),
                                                               trainable=main.autoencoder_bool),
                                         trainable=main.autoencoder_bool)(x)

        if main.autoencoder_unet_bool:
            res_connections_x = res_connections.pop()

            if main.autoencoder_unet_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x, res_connections_x])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                       kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                       False, gaussian_stddev, dropout_amount)
            else:
                x = k.layers.Add(trainable=main.autoencoder_bool)([x, res_connections_x])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

        for i in range(layer_2_layers):
            x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_2_depth,
                                                       (layer_2_kernel_size, layer_2_kernel_size, layer_2_kernel_size),
                                                       (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev,
                                                       dropout_amount, main.autoencoder_resnet_bool,
                                                       main.autoencoder_bool, main.autoencoder_resnet_concatenate_bool,
                                                       main.autoencoder_densenet_bool)

    # layer 1
    if main.cluster_bool:
        if main.up_stride_bool:
            if main.up_upsample_too_bool:
                x_1 = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                             layer_1_depth, (layer_1_kernel_size,
                                                                             layer_1_kernel_size,
                                                                             layer_1_kernel_size), (2, 2, 2),
                                                             ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                                             main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                             main.autoencoder_resnet_concatenate_bool,
                                                             main.autoencoder_densenet_bool)

                x_2 = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 2),
                                                               trainable=main.autoencoder_bool),
                                         trainable=main.autoencoder_bool)(x)

                if main.down_max_pool_too_concatenate_bool:
                    concatenate_depth = x.shape[-1]

                    x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                   trainable=main.autoencoder_bool)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)

                    kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                    x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool,
                                                           grouped_channel_shuffle_bool, kernel_groups,
                                                           main.autoencoder_bool, ltwo_weight, lone_weight,
                                                           False, gaussian_stddev, dropout_amount)
                else:
                    if x_2.shape != x_1.shape:
                        if gaussian_stddev > 0.0:
                            x_2 = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                         trainable=main.autoencoder_bool)(x_2)

                        x_2 = k.layers.ReLU(max_value=6.0,
                                            negative_slope=0.2)(x_2)

                        kernel_groups = layers.get_kernel_groups(x_2, grouped_bool, x_1.shape[-1])

                        x_2 = layers.bottleneck_conv3D_transpose(x_2, x_1.shape[-1], grouped_bool,
                                                                 grouped_channel_shuffle_bool,
                                                                 kernel_groups, main.autoencoder_bool, ltwo_weight,
                                                                 lone_weight,
                                                                 False, gaussian_stddev, dropout_amount)

                    x = k.layers.Add(trainable=main.autoencoder_bool)([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                                   trainable=main.autoencoder_bool)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)
            else:
                x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                           layer_1_depth, (layer_1_kernel_size,
                                                                           layer_1_kernel_size,
                                                                           layer_1_kernel_size), (2, 2, 2), ltwo_weight,
                                                           lone_weight, gaussian_stddev, dropout_amount,
                                                           main.autoencoder_resnet_bool, main.autoencoder_bool,
                                                           main.autoencoder_resnet_concatenate_bool,
                                                           main.autoencoder_densenet_bool)
        else:
            x = k.layers.TimeDistributed(k.layers.UpSampling3D(size=(2, 2, 2),
                                                               trainable=main.autoencoder_bool),
                                         trainable=main.autoencoder_bool)(x)

        if main.autoencoder_unet_bool:
            res_connections_x = res_connections.pop()

            if main.autoencoder_unet_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate(trainable=main.autoencoder_bool)([x, res_connections_x])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                       kernel_groups, main.autoencoder_bool, ltwo_weight, lone_weight,
                                                       False, gaussian_stddev, dropout_amount)
            else:
                x = k.layers.Add(trainable=main.autoencoder_bool)([x, res_connections_x])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev,
                                               trainable=main.autoencoder_bool)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

        for i in range(layer_1_layers):
            x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_1_depth,
                                                       (layer_1_kernel_size, layer_1_kernel_size, layer_1_kernel_size),
                                                       (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev,
                                                       dropout_amount, main.autoencoder_resnet_bool,
                                                       main.autoencoder_bool, main.autoencoder_resnet_concatenate_bool,
                                                       main.autoencoder_densenet_bool)

    current_depth = x.shape[-1]

    while current_depth > 2:

        current_depth = current_depth / 2

        kernel_groups = layers.get_kernel_groups(x, grouped_bool, current_depth)

        x = layers.bottleneck_conv3D(x, current_depth, grouped_bool, grouped_channel_shuffle_bool, kernel_groups,
                                     main.autoencoder_bool, ltwo_weight, lone_weight, False, gaussian_stddev,
                                     dropout_amount)

    # output
    x = k.layers.TimeDistributed(layers.ReflectionPadding3D(padding=(1, 1, 1),
                                                            trainable=main.autoencoder_bool),
                                 trainable=main.autoencoder_bool)(x)
    x = k.layers.TimeDistributed(k.layers.Conv3D(filters=1,
                                                 kernel_size=(3, 3, 3),
                                                 strides=(1, 1, 1),
                                                 dilation_rate=(1, 1, 1),
                                                 groups=1,
                                                 padding="valid",
                                                 kernel_initializer="glorot_normal",
                                                 bias_initializer=k.initializers.Constant(0.0),
                                                 trainable=main.autoencoder_bool),
                                 trainable=main.autoencoder_bool,
                                 name="output_2")(x)

    return x


def get_tensors(input_shape, grouped_bool, grouped_channel_shuffle_bool, input_gaussian_stddev, gaussian_stddev,
                ltwo_weight, lone_weight, dropout_amount, latent_size, output_layer_size):
    print("get_tensors")

    x, input_x = get_input(input_shape)

    res_connections = []

    if not main.pca_only_classifier_bool:
        x, res_connections = get_encoder(x, grouped_bool, grouped_channel_shuffle_bool, input_gaussian_stddev,
                                         gaussian_stddev, ltwo_weight, lone_weight, dropout_amount)

    x, x_latent_center, mu, sigma = get_latent(x, grouped_bool, latent_size, ltwo_weight, lone_weight, gaussian_stddev,
                                               dropout_amount)

    output_1_x = get_classifier(x_latent_center, output_layer_size, ltwo_weight, lone_weight, dropout_amount,
                                gaussian_stddev)

    output_2_x = get_decoder(x, grouped_bool, grouped_channel_shuffle_bool, res_connections, ltwo_weight, lone_weight,
                             gaussian_stddev, dropout_amount)

    return input_x, output_1_x, output_2_x, mu, sigma


def get_model(input_shape, output_layer_size):
    print("get_model")

    grouped_bool = parameters.grouped_bool
    grouped_channel_shuffle_bool = parameters.grouped_channel_shuffle_bool

    input_gaussian_stddev = 0.0
    gaussian_stddev = parameters.gaussian_stddev
    ltwo_weight = 0.0
    lone_weight = parameters.lone_weight

    # use 0.5 when training for real
    dropout_amount = parameters.dropout_amount
    beta = parameters.beta

    latent_size = parameters.latent_size

    input_x, output_1_x, output_2_x, mu, sigma = get_tensors(input_shape, grouped_bool, grouped_channel_shuffle_bool,
                                                             input_gaussian_stddev, gaussian_stddev, ltwo_weight,
                                                             lone_weight, dropout_amount, latent_size,
                                                             output_layer_size)

    if main.autoencoder_bool:
        if main.classifier_bool:
            loss_weights = [1.0 / 0.5, 1.0 / 1700.0]
        else:
            loss_weights = [0.0, 1.0]
    else:
        if main.classifier_bool:
            loss_weights = [1.0, 0.0]
        else:
            loss_weights = [0.0, 0.0]

    model = k.Model(inputs=input_x, outputs=[output_1_x, output_2_x])

    if main.cnn_only_autoencoder_bool:
        model.compile(optimizer=k.optimizers.Nadam(learning_rate=main.initial_learning_rate,
                                                   global_clipnorm=1.0,
                                                   clipvalue=1.0),
                      loss={"output_1": k.losses.log_cosh,
                            "output_2": k.losses.log_cosh},
                      loss_weights=loss_weights,
                      metrics=[loss.accuracy_correlation_coefficient])
    else:
        if main.dv_bool:
            model.compile(optimizer=k.optimizers.Nadam(learning_rate=main.initial_learning_rate,
                                                       global_clipnorm=1.0,
                                                       clipvalue=1.0),
                          loss={"output_1": loss.kl_log_cosh(mu, sigma, beta),
                                "output_2": loss.kl_log_cosh(mu, sigma, beta)},
                          loss_weights=loss_weights,
                          metrics=[loss.accuracy_correlation_coefficient])
        else:
            model.compile(optimizer=k.optimizers.Nadam(learning_rate=main.initial_learning_rate,
                                                       global_clipnorm=1.0,
                                                       clipvalue=1.0),
                          loss={"output_1": k.losses.log_cosh,
                                "output_2": k.losses.log_cosh},
                          loss_weights=loss_weights,
                          metrics=[loss.accuracy_correlation_coefficient])

    return model
