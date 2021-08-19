# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


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

    # layer_depth = [16, 32, 64]
    layer_depth = []
    layer_kernel_size = [3, 3, 3]
    layer_layers = [2, 2, 2]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

    if input_gaussian_stddev > 0.0:
        x = k.layers.GaussianNoise(stddev=input_gaussian_stddev)(x)

    res_connections = []

    # layer 1
    for i in range(len(layer_depth)):
        for j in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_depth[i],
                                             (layer_kernel_size[i], layer_kernel_size[i], layer_kernel_size[i]),
                                             (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                             main.autoencoder_resnet_bool, main.autoencoder_resnet_concatenate_bool,
                                             main.autoencoder_densenet_bool)

        if gaussian_stddev > 0.0:
            res_connections.append(k.layers.GaussianNoise(stddev=gaussian_stddev)(x))
        else:
            res_connections.append(x)

        if main.down_stride_bool:
            if main.down_max_pool_too_bool:
                x_1 = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_depth[i],
                                                   (layer_kernel_size[i], layer_kernel_size[i], layer_kernel_size[i]),
                                                   layer_stride[i], ltwo_weight, lone_weight, gaussian_stddev,
                                                   dropout_amount, main.autoencoder_resnet_bool,
                                                   main.autoencoder_resnet_concatenate_bool,
                                                   main.autoencoder_densenet_bool)

                x_2 = k.layers.AveragePooling3D(pool_size=(layer_kernel_size[i],
                                                           layer_kernel_size[i],
                                                           layer_kernel_size[i]),
                                                strides=layer_stride[i],
                                                padding="same")(x)

                if main.down_max_pool_too_concatenate_bool:
                    concatenate_depth = x.shape[-1]

                    x = k.layers.Concatenate()([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)

                    kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                    x = layers.bottleneck_conv3D(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                 kernel_groups, ltwo_weight, lone_weight, False, gaussian_stddev,
                                                 dropout_amount)
                else:
                    x = k.layers.Add()([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)
            else:
                x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_depth[i],
                                                 (layer_kernel_size[i], layer_kernel_size[i], layer_kernel_size[i]),
                                                 layer_stride[i], ltwo_weight, lone_weight, gaussian_stddev,
                                                 dropout_amount, main.autoencoder_resnet_bool,
                                                 main.autoencoder_resnet_concatenate_bool,
                                                 main.autoencoder_densenet_bool)
        else:
            x = k.layers.MaxPooling3D(pool_size=layer_stride[i], padding="same")(x)

    return x, res_connections


def get_latent(x, grouped_bool, grouped_channel_shuffle_bool, gaussian_stddev, ltwo_weight, lone_weight,
               dropout_amount):
    print("get_latent")

    layer_depth = [8]
    layer_kernel_size = [3]
    layer_layers = [2]

    # layer 1
    for i in range(len(layer_depth)):
        for j in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_depth[i],
                                             (layer_kernel_size[i], layer_kernel_size[i], layer_kernel_size[i]),
                                             (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                             main.autoencoder_resnet_bool, main.autoencoder_resnet_concatenate_bool,
                                             main.autoencoder_densenet_bool)

    return x


def get_decoder(x, grouped_bool, grouped_channel_shuffle_bool, res_connections, ltwo_weight, lone_weight,
                gaussian_stddev, dropout_amount):
    print("get_decoder")

    # layer_depth = [64, 32, 16]
    layer_depth = []
    layer_kernel_size = [3, 3, 3]
    layer_layers = [2, 2, 2]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

    for i in range(len(layer_depth)):
        if main.up_stride_bool:
            if main.up_upsample_too_bool:
                x_1 = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                             layer_depth[i], (layer_kernel_size[i],
                                                                              layer_kernel_size[i],
                                                                              layer_kernel_size[i]), layer_stride[i],
                                                             ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                                             main.autoencoder_resnet_bool,
                                                             main.autoencoder_resnet_concatenate_bool,
                                                             main.autoencoder_densenet_bool)

                x_2 = k.layers.UpSampling3D(size=layer_stride[i])(x)

                if main.down_max_pool_too_concatenate_bool:
                    concatenate_depth = x.shape[-1]

                    x = k.layers.Concatenate()([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)

                    kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                    x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool,
                                                           grouped_channel_shuffle_bool, kernel_groups,
                                                           ltwo_weight, lone_weight, False, gaussian_stddev,
                                                           dropout_amount)
                else:
                    if x_2.shape != x_1.shape:
                        if gaussian_stddev > 0.0:
                            x_2 = k.layers.GaussianNoise(stddev=gaussian_stddev)(x_2)

                        x_2 = k.layers.ReLU(max_value=6.0,
                                            negative_slope=0.2)(x_2)

                        kernel_groups = layers.get_kernel_groups(x_2, grouped_bool, x_1.shape[-1])

                        x_2 = layers.bottleneck_conv3D_transpose(x_2, x_1.shape[-1], grouped_bool,
                                                                 grouped_channel_shuffle_bool,
                                                                 kernel_groups, ltwo_weight,
                                                                 lone_weight,
                                                                 False, gaussian_stddev, dropout_amount)

                    x = k.layers.Add()([x_1, x_2])

                    if gaussian_stddev > 0.0:
                        x = k.layers.GaussianNoise(stddev=gaussian_stddev)(x)

                    x = k.layers.ReLU(max_value=6.0,
                                      negative_slope=0.2)(x)
            else:
                x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool,
                                                           layer_depth[i], (layer_kernel_size[i],
                                                                            layer_kernel_size[i],
                                                                            layer_kernel_size[i]), layer_stride[i],
                                                           ltwo_weight, lone_weight, gaussian_stddev, dropout_amount,
                                                           main.autoencoder_resnet_bool,
                                                           main.autoencoder_resnet_concatenate_bool,
                                                           main.autoencoder_densenet_bool)
        else:
            x = k.layers.UpSampling3D(size=layer_stride[i])(x)

        if main.autoencoder_unet_bool:
            res_connections_x = res_connections.pop()

            if x.shape != res_connections_x.shape:
                kernel_groups = layers.get_kernel_groups(x, grouped_bool, res_connections_x.shape[-1])

                x = layers.bottleneck_conv3D_transpose(x, res_connections_x.shape[-1], grouped_bool,
                                                       grouped_channel_shuffle_bool,
                                                       kernel_groups, ltwo_weight,
                                                       lone_weight,
                                                       False, gaussian_stddev, dropout_amount)

            if main.autoencoder_unet_concatenate_bool:
                concatenate_depth = x.shape[-1]

                x = k.layers.Concatenate()([x, res_connections_x])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

                kernel_groups = layers.get_kernel_groups(x, grouped_bool, concatenate_depth)

                x = layers.bottleneck_conv3D_transpose(x, concatenate_depth, grouped_bool, grouped_channel_shuffle_bool,
                                                       kernel_groups, ltwo_weight, lone_weight, False, gaussian_stddev,
                                                       dropout_amount)
            else:
                x = k.layers.Add()([x, res_connections_x])

                if gaussian_stddev > 0.0:
                    x = k.layers.GaussianNoise(stddev=gaussian_stddev)(x)

                x = k.layers.ReLU(max_value=6.0,
                                  negative_slope=0.2)(x)

        for j in range(layer_layers[i]):
            x = layers.get_transpose_convolution_layer(x, grouped_bool, grouped_channel_shuffle_bool, layer_depth[i],
                                                       (layer_kernel_size[i],
                                                        layer_kernel_size[i],
                                                        layer_kernel_size[i]),
                                                       (1, 1, 1), ltwo_weight, lone_weight, gaussian_stddev,
                                                       dropout_amount, main.autoencoder_resnet_bool,
                                                       main.autoencoder_resnet_concatenate_bool,
                                                       main.autoencoder_densenet_bool)

    # output
    x = layers.ReflectionPadding3D(padding=(1, 1, 1))(x)
    x = k.layers.Conv3D(filters=1,
                        kernel_size=(3, 3, 3),
                        strides=(1, 1, 1),
                        dilation_rate=(1, 1, 1),
                        groups=1,
                        padding="valid",
                        kernel_initializer="glorot_normal",
                        bias_initializer=k.initializers.Constant(0.0),
                        name="output")(x)

    return x


def get_tensors(input_shape, grouped_bool, grouped_channel_shuffle_bool, input_gaussian_stddev, gaussian_stddev,
                ltwo_weight, lone_weight, dropout_amount):
    print("get_tensors")

    x, input_x = get_input(input_shape)

    x, res_connections = get_encoder(x, grouped_bool, grouped_channel_shuffle_bool, input_gaussian_stddev,
                                     gaussian_stddev, ltwo_weight, lone_weight, dropout_amount)

    x = get_latent(x, grouped_bool, grouped_channel_shuffle_bool, gaussian_stddev, ltwo_weight, lone_weight,
                   dropout_amount)

    output_x = get_decoder(x, grouped_bool, grouped_channel_shuffle_bool, res_connections, ltwo_weight, lone_weight,
                           gaussian_stddev, dropout_amount)

    return input_x, output_x


def get_model(input_shape):
    print("get_model")

    grouped_bool = parameters.grouped_bool
    grouped_channel_shuffle_bool = parameters.grouped_channel_shuffle_bool

    input_gaussian_stddev = 0.0
    gaussian_stddev = parameters.gaussian_stddev
    ltwo_weight = 0.0
    lone_weight = parameters.lone_weight

    # use 0.5 when training for real
    dropout_amount = parameters.dropout_amount

    input_x, output_x = get_tensors(input_shape, grouped_bool, grouped_channel_shuffle_bool,
                                    input_gaussian_stddev, gaussian_stddev, ltwo_weight, lone_weight, dropout_amount)

    model = k.Model(inputs=input_x, outputs=[output_x])

    model.compile(optimizer=k.optimizers.Nadam(global_clipnorm=1.0, clipvalue=1.0),
                  loss={"output": k.losses.mean_squared_error}, loss_weights=[1.0],
                  metrics=[loss.accuracy_correlation_coefficient])

    return model
