# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import gc
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as k


import main

if main.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(main.seed_value)


import parameters
import layers
import losses


def get_input(input_shape):
    print("get_input")

    x = k.layers.Input(input_shape)

    return x, x


def get_encoder(x):
    print("get_encoder")

    layer_layers = [2, 2, 2, 2, 2]
    layer_depth = [2, 4, 8, 16, 32]
    layer_kernel_size = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    layer_groups = [1, 1, 1, 1, 1]

    unet_connections = []

    x = k.layers.Conv3D(filters=1,
                        kernel_size=(1, 1, 1),
                        strides=(1, 1, 1),
                        dilation_rate=(1, 1, 1),
                        groups=1,
                        padding="valid",
                        kernel_initializer=k.initializers.Constant(1.0),
                        bias_initializer=k.initializers.Constant(0.0),
                        trainable=False,
                        name="input")(x)

    x = layers.get_gaussian_noise(x, parameters.input_gaussian_sigma)

    x = layers.get_convolution_layer(x, 1, layer_kernel_size[0], (1, 1, 1), 1)

    for i in range(len(layer_layers)):
        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i])

        unet_connections.append(layers.get_gaussian_noise(x, parameters.skip_gaussian_sigma))

        x1 = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], layer_stride[i], layer_groups[i])

        x2 = layers.get_reflection_padding(x, layer_kernel_size[i])
        x2 = k.layers.MaxPooling3D(pool_size=layer_kernel_size[i],
                                   strides=layer_stride[i],
                                   padding="valid")(x2)

        x = k.layers.Concatenate()([x1, x2])

    return x, unet_connections


def get_latent(x):
    print("get_latent")

    layer_layers = [1]
    layer_depth = [64]
    layer_kernel_size = [(3, 3, 3)]
    layer_groups = [1]

    for i in range(len(layer_layers)):
        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i])

        x = k.layers.Conv3D(filters=layer_depth[i],
                            kernel_size=(1, 1, 1),
                            strides=(1, 1, 1),
                            dilation_rate=(1, 1, 1),
                            groups=layer_depth[i],
                            padding="valid",
                            kernel_initializer=k.initializers.Constant(1.0),
                            bias_initializer=k.initializers.Constant(0.0),
                            trainable=False,
                            name="latent")(x)

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i])

    return x


def get_decoder(x, unet_connections):
    print("get_decoder")

    layer_layers = [2, 2, 2, 2, 2]
    layer_depth = [32, 16, 8, 4, 2]
    layer_kernel_size = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    layer_groups = [1, 1, 1, 1, 1]

    for i in range(len(layer_depth)):
        x = layers.get_transpose_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i])

        x = k.layers.UpSampling3D(size=tuple([x * 2 for x in layer_stride[i]]))(x)

        x = layers.get_reflection_padding(x, layer_kernel_size[i])
        x = k.layers.AveragePooling3D(pool_size=layer_kernel_size[i],
                                      strides=layer_stride[i],
                                      padding="valid")(x)

        x = k.layers.Concatenate()([x, unet_connections.pop()])

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i])

    x = layers.get_convolution_layer(x, 1, layer_kernel_size[-1], (1, 1, 1), 1)

    x = k.layers.Conv3D(filters=1,
                        kernel_size=(1, 1, 1),
                        strides=(1, 1, 1),
                        dilation_rate=(1, 1, 1),
                        groups=1,
                        padding="valid",
                        kernel_initializer=k.initializers.Constant(1.0),
                        bias_initializer=k.initializers.Constant(0.0),
                        trainable=False,
                        name="output")(x)

    return x


def get_tensors(input_shape):
    print("get_tensors")

    x, input_x = get_input(input_shape)

    x, unet_connections = get_encoder(x)
    x = get_latent(x)
    output_x = get_decoder(x, unet_connections)

    return input_x, output_x


def get_model_all(input_shape):
    print("get_model_all")

    model = get_model(input_shape)

    loss = get_loss()
    optimiser = get_optimiser()

    gc.collect()
    k.backend.clear_session()

    return model, optimiser, loss


def get_model(input_shape):
    print("get_model")

    input_x, output_x = get_tensors(input_shape)

    model = k.Model(inputs=input_x, outputs=[output_x])

    gc.collect()
    k.backend.clear_session()

    return model


def get_optimiser():
    print("get_optimiser")

    if parameters.weight_decay > 0.0:
        optimiser = tfa.optimizers.extend_with_decoupled_weight_decay(tf.keras.optimizers.Nadam)(weight_decay=parameters.weight_decay, clipvalue=6.0)  # noqa
    else:
        optimiser = tf.keras.optimizers.Nadam(clipvalue=6.0)

    return optimiser


def get_loss():
    print("get_loss")

    if parameters.relative_difference_bool:
        loss = losses.log_cosh_relative_difference_loss
    else:
        loss = losses.log_cosh_loss

    return loss
