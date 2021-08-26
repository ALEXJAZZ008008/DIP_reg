# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow.keras as k
from tqdm import trange

import main
import parameters
import layers
import losses


if main.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(main.seed_value)


def get_input(input_shape):
    print("get_input")

    input_x = k.layers.Input(input_shape)

    return input_x, input_x


def get_encoder(x, kernel_regularisation, sparseness):
    print("get_encoder")

    layer_layers = [2, 2, 2, 2]
    layer_depth = [4, 8, 16, 32]
    layer_kernel_size = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    layer_groups = [2, 2, 2, 2]

    unet_connections = []

    x = k.layers.GaussianNoise(parameters.gaussian_sigma)(x)

    x = k.layers.Conv3D(filters=layer_depth[0],
                        kernel_size=(1, 1, 1),
                        strides=(1, 1, 1),
                        dilation_rate=(1, 1, 1),
                        groups=1,
                        padding="valid",
                        kernel_initializer="he_normal",
                        bias_initializer=k.initializers.Constant(0.0))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU(alpha_initializer=k.initializers.Constant(0.3),
                       shared_axes=[1, 2, 3])(x)

    # layer 1
    for i in trange(len(layer_layers)):
        for _ in trange(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i],
                                             kernel_regularisation, sparseness)

        unet_connections.append(k.layers.GaussianNoise(parameters.gaussian_sigma)(x))

        x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], layer_stride[i], layer_groups[i],
                                         kernel_regularisation, sparseness)

    return x, unet_connections


def get_latent(x, kernel_regularisation, sparseness):
    print("get_latent")

    layer_layers = [2]
    layer_depth = [64]
    layer_kernel_size = [(3, 3, 3)]
    layer_groups = [2]

    # layer 1
    for i in trange(len(layer_layers)):
        x = layers.get_reflection_padding(x, layer_kernel_size[i])
        x = k.layers.Conv3D(filters=layer_depth[i],
                            kernel_size=layer_kernel_size[i],
                            strides=(1, 1, 1),
                            dilation_rate=(1, 1, 1),
                            groups=layer_groups[i],
                            padding="valid",
                            kernel_initializer="he_normal",
                            bias_initializer=k.initializers.Constant(0.0),
                            name="latent")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU(alpha_initializer=k.initializers.Constant(0.3),
                           shared_axes=[1, 2, 3])(x)
        x = k.layers.Lambda(layers.channel_shuffle, arguments={"groups": layer_groups[i]})(x)

        x = k.layers.GaussianNoise(parameters.gaussian_sigma)(x)

        for _ in trange(layer_layers[i] - 1):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i],
                                             kernel_regularisation, sparseness)

    return x


def get_decoder(x, unet_connections, kernel_regularisation, sparseness):
    print("get_decoder")

    layer_layers = [2, 2, 2, 2]
    layer_depth = [32, 16, 8, 4]
    layer_kernel_size = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    layer_groups = [2, 2, 2, 2]

    for i in trange(len(layer_depth)):
        x = layers.get_transpose_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i],
                                                   kernel_regularisation, sparseness)
        x = k.layers.UpSampling3D(size=layer_stride[i])(x)

        x = k.layers.Concatenate()([x, unet_connections.pop()])

        for _ in trange(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i],
                                             kernel_regularisation, sparseness)

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

    kernel_regularisation = 0.0
    sparseness = 0.0

    x, input_x = get_input(input_shape)

    x, unet_connections = get_encoder(x, kernel_regularisation, sparseness)

    x = get_latent(x, kernel_regularisation, sparseness)

    output_x = get_decoder(x, unet_connections, kernel_regularisation, sparseness)

    return input_x, output_x


def get_model(input_shape):
    print("get_model")

    input_x, output_x = get_tensors(input_shape)

    model = k.Model(inputs=input_x, outputs=[output_x])

    # optimiser = tfa.optimizers.extend_with_decoupled_weight_decay(tf.keras.optimizers.Nadam)\
    #     (weight_decay=0.0, learning_rate=1e-02, clipvalue=1.0) # noqa

    optimiser = tf.optimizers.Nadam(learning_rate=1e-03, clipvalue=1.0)

    if parameters.relative_difference_bool:
        # loss = losses.log_cosh_total_variation_loss
        loss = losses.log_cosh_loss
    else:
        loss = losses.log_cosh_loss

    return model, optimiser, loss
