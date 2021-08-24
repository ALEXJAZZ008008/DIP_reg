# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow.keras as k
from tqdm import trange

import layers
import loss
import parameters


def get_input(input_shape):
    print("get_input")

    input_x = k.layers.Input(input_shape)

    return input_x, input_x


def get_encoder(x, kernel_weight, activity_sparseness):
    print("get_encoder")

    layer_layers = [2, 2, 2, 2]
    layer_depth = [4, 8, 16, 32]
    layer_kernel_size = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    layer_groups = [2, 2, 2, 2]

    unet_connections = []

    x = k.layers.GaussianNoise(parameters.gaussian_sigma)(x)

    x = layers.get_convolution_layer(x, layer_depth[0], layer_kernel_size[0], (1, 1, 1), 1, kernel_weight,
                                     activity_sparseness)

    # layer 1
    for i in trange(len(layer_layers)):
        for _ in trange(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i],
                                             kernel_weight, activity_sparseness)

        unet_connections.append(x)

        x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], layer_stride[i], layer_groups[i],
                                         kernel_weight, activity_sparseness)

    return x, unet_connections


def get_latent(x, kernel_weight, activity_sparseness):
    print("get_latent")

    layer_layers = [2]
    layer_depth = [64]
    layer_kernel_size = [(3, 3, 3)]
    layer_groups = [2]

    # layer 1
    for i in trange(len(layer_layers)):
        for _ in trange(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i],
                                             kernel_weight, activity_sparseness)

    return x


def get_decoder(x, unet_connections, kernel_weight, activity_sparseness):
    print("get_decoder")

    layer_layers = [2, 2, 2, 2]
    layer_depth = [32, 16, 8, 4]
    layer_kernel_size = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    layer_groups = [2, 2, 2, 2]

    for i in trange(len(layer_depth)):
        x = layers.get_transpose_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i],
                                                   kernel_weight, activity_sparseness)
        x = k.layers.UpSampling3D(size=layer_stride[i])(x)

        unet_connection_x = unet_connections.pop()

        x = k.layers.Add()([x, unet_connection_x])

        for _ in trange(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i],
                                             kernel_weight, activity_sparseness)

    # output
    output_kernel_size = (3, 3, 3)

    x = layers.get_reflection_padding(x, output_kernel_size)
    x = k.layers.Conv3D(filters=1,
                        kernel_size=output_kernel_size,
                        strides=(1, 1, 1),
                        dilation_rate=(1, 1, 1),
                        groups=1,
                        padding="valid",
                        kernel_initializer="he_normal",
                        bias_initializer=k.initializers.Constant(0.0),
                        name="output")(x)

    return x


def get_tensors(input_shape):
    print("get_tensors")

    kernel_weight = 0.0
    activity_sparseness = 0.0

    x, input_x = get_input(input_shape)

    x, unet_connections = get_encoder(x, kernel_weight, activity_sparseness)

    x = get_latent(x, kernel_weight, activity_sparseness)

    output_x = get_decoder(x, unet_connections, kernel_weight, activity_sparseness)

    return input_x, output_x


def get_model(input_shape):
    print("get_model")

    input_x, output_x = get_tensors(input_shape)

    model = k.Model(inputs=input_x, outputs=[output_x])

    if parameters.relative_difference_bool:
        model.compile(optimizer=k.optimizers.Nadam(clipvalue=6.0),
                      loss={"output": loss.log_cosh_total_variation_loss}, loss_weights=[1.0],
                      metrics=[loss.correlation_coefficient_accuracy])
    else:
        model.compile(optimizer=k.optimizers.Nadam(clipvalue=6.0),
                      loss={"output": loss.log_cosh_loss}, loss_weights=[1.0],
                      metrics=[loss.correlation_coefficient_accuracy])

    return model
