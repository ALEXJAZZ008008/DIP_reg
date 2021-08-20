# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow.keras as k

import main
import layers
import loss
import parameters


def get_input(input_shape):
    print("get_input")

    input_x = k.layers.Input(input_shape)

    return input_x, input_x


def get_encoder(x):
    print("get_encoder")

    layer_depth = [16, 32, 64]
    layer_kernel_size = [3, 3, 3]
    layer_layers = [2, 2, 2]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

    res_connections = []

    # layer 1
    for i in range(len(layer_depth)):
        for j in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i],
                                             (layer_kernel_size[i], layer_kernel_size[i], layer_kernel_size[i]),
                                             (1, 1, 1))

        res_connections.append(x)

        if main.down_stride_bool:
            x = layers.get_convolution_layer(x, layer_depth[i],
                                             (layer_kernel_size[i], layer_kernel_size[i], layer_kernel_size[i]),
                                             layer_stride[i])
        else:
            x = k.layers.MaxPooling3D(pool_size=layer_stride[i], padding="same")(x)

    return x, res_connections


def get_latent(x):
    print("get_latent")

    layer_depth = [128]
    layer_kernel_size = [3]
    layer_layers = [2]

    # layer 1
    for i in range(len(layer_depth)):
        for j in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i],
                                             (layer_kernel_size[i], layer_kernel_size[i], layer_kernel_size[i]),
                                             (1, 1, 1))

    return x


def get_decoder(x, res_connections):
    print("get_decoder")

    layer_depth = [64, 32, 16]
    layer_kernel_size = [3, 3, 3]
    layer_layers = [2, 2, 2]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

    for i in range(len(layer_depth)):
        if main.up_stride_bool:
            x = layers.get_transpose_convolution_layer(x, layer_depth[i],
                                                       (layer_kernel_size[i],
                                                        layer_kernel_size[i],
                                                        layer_kernel_size[i]), layer_stride[i])
        else:
            x = k.layers.Conv3DTranspose(filters=layer_depth[i],
                                         kernel_size=(layer_kernel_size[i], layer_kernel_size[i], layer_kernel_size[i]),
                                         strides=(1, 1, 1),
                                         dilation_rate=1,
                                         groups=1,
                                         padding="same",
                                         kernel_initializer="he_normal",
                                         bias_initializer=k.initializers.Constant(0.0))(x)

            x = k.layers.UpSampling3D(size=layer_stride[i])(x)

        if main.autoencoder_unet_bool:
            res_connections_x = res_connections.pop()

            if main.autoencoder_unet_concatenate_bool:
                x = k.layers.Concatenate()([x, res_connections_x])
            else:
                x = k.layers.Add()([x, res_connections_x])

        for j in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i],
                                             (layer_kernel_size[i], layer_kernel_size[i], layer_kernel_size[i]),
                                             (1, 1, 1))

    # output
    x = k.layers.Conv3D(filters=1,
                        kernel_size=(3, 3, 3),
                        strides=(1, 1, 1),
                        dilation_rate=(1, 1, 1),
                        groups=1,
                        padding="same",
                        kernel_initializer="he_normal",
                        bias_initializer=k.initializers.Constant(0.0),
                        name="output")(x)

    return x


def get_tensors(input_shape):
    print("get_tensors")

    x, input_x = get_input(input_shape)

    x, res_connections = get_encoder(x)

    x = get_latent(x)

    output_x = get_decoder(x, res_connections)

    return input_x, output_x


def get_model(input_shape):
    print("get_model")

    input_x, output_x = get_tensors(input_shape)

    model = k.Model(inputs=input_x, outputs=[output_x])

    if parameters.total_variation_bool:
        model.compile(optimizer=k.optimizers.Adam(),
                      loss={"output": loss.mean_square_error_total_variation_loss}, loss_weights=[1.0],
                      metrics=[loss.accuracy_correlation_coefficient])
    else:
        model.compile(optimizer=k.optimizers.Adam(),
                      loss={"output": loss.mean_squared_error}, loss_weights=[1.0],
                      metrics=[loss.accuracy_correlation_coefficient])

    return model
