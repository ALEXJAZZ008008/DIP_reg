# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import gc
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as k


import DIP_RDP

if DIP_RDP.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(DIP_RDP.seed_value)


import parameters
import layers
import losses


def get_input(input_shape):
    print("get_input")

    x = k.layers.Input(input_shape)

    return x, x


def get_encoder(x):
    print("get_encoder")

    layer_layers = parameters.layer_layers[:-1]
    layer_depth = parameters.layer_depth[:-1]
    layer_groups = parameters.layer_groups[:-1]

    if not isinstance(layer_layers, list):
        layer_layers = [layer_layers]

    if not isinstance(layer_depth, list):
        layer_depth = [layer_depth]

    if not isinstance(layer_groups, list):
        layer_groups = [layer_groups]

    unet_connections = []

    x = k.layers.Conv3D(filters=1,
                        kernel_size=(1, 1, 1),
                        strides=(1, 1, 1),
                        dilation_rate=(1, 1, 1),
                        groups=1,
                        padding="same",
                        kernel_initializer=k.initializers.Constant(1.0),
                        bias_initializer=k.initializers.Constant(0.0),
                        trainable=False,
                        name="input")(x)

    x = layers.get_gaussian_noise(x, parameters.input_gaussian_sigma)

    for i in range(len(layer_layers)):
        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i])

        unet_connections.append(layers.get_gaussian_noise(x, parameters.skip_gaussian_sigma))

        x1 = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (2, 2, 2), layer_groups[i])

        x2 = k.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                   strides=(2, 2, 2),
                                   padding="valid")(x)

        x = k.layers.Concatenate()([x1, x2])

    return x, unet_connections


def get_latent(x):
    print("get_latent")

    layer_layers = [parameters.layer_layers[-1]]
    layer_depth = [parameters.layer_depth[-1]]
    layer_groups = [parameters.layer_groups[-1]]

    latnet_x = x

    for i in range(len(layer_layers)):
        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i])

        latnet_x = k.layers.Conv3D(filters=layer_depth[i],
                                   kernel_size=(1, 1, 1),
                                   strides=(1, 1, 1),
                                   dilation_rate=(1, 1, 1),
                                   groups=layer_depth[i],
                                   padding="same",
                                   kernel_initializer=k.initializers.Constant(1.0),
                                   bias_initializer=k.initializers.Constant(0.0),
                                   trainable=False,
                                   name="latent")(x)

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i])

    return x, latnet_x


def get_decoder(x, unet_connections):
    print("get_decoder")

    layer_layers = parameters.layer_layers[:-1]
    layer_depth = parameters.layer_depth[:-1]
    layer_groups = parameters.layer_groups[:-1]

    if not isinstance(layer_layers, list):
        layer_layers = [layer_layers]

    if not isinstance(layer_depth, list):
        layer_depth = [layer_depth]

    if not isinstance(layer_groups, list):
        layer_groups = [layer_groups]

    layer_layers.reverse()
    layer_depth.reverse()
    layer_groups.reverse()

    for i in range(len(layer_depth)):
        x = layers.get_transpose_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i])

        x = k.layers.UpSampling3D(size=tuple([x * 2 for x in (2, 2, 2)]))(x)

        x = k.layers.AveragePooling3D(pool_size=(2, 2, 2),
                                      strides=(2, 2, 2),
                                      padding="valid")(x)

        x = k.layers.Concatenate()([x, unet_connections.pop()])

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i])

    x = layers.get_convolution_layer(x, 1, (3, 3, 3), (1, 1, 1), 1)

    x = k.layers.Conv3D(filters=1,
                        kernel_size=(1, 1, 1),
                        strides=(1, 1, 1),
                        dilation_rate=(1, 1, 1),
                        groups=1,
                        padding="same",
                        kernel_initializer=k.initializers.Constant(1.0),
                        bias_initializer=k.initializers.Constant(0.0),
                        trainable=False,
                        name="output")(x)

    return x


def get_tensors(input_shape):
    print("get_tensors")

    x, input_x = get_input(input_shape)

    x, unet_connections = get_encoder(x)
    x, latnet_x = get_latent(x)
    output_x = get_decoder(x, unet_connections)

    return input_x, latnet_x, output_x


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

    input_x, latnet_x, output_x = get_tensors(input_shape)

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
        if parameters.total_variation_bool:
            loss = losses.log_cosh_total_variation_loss
        else:
            loss = losses.log_cosh_loss

    return loss
