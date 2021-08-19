# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf
import tensorflow.keras as k


# https://stackoverflow.com/questions/42665359/how-do-you-compute-accuracy-in-a-regression-model-after-rounding-predictions-to
def accuracy_epsilon(y_true, y_pred):
    return k.backend.mean(k.backend.abs(y_true - y_pred) < 1e-04)


# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred

    mx = k.backend.mean(x)
    my = k.backend.mean(y)

    xm = x - mx
    ym = y - my

    r_num = k.backend.sum(tf.multiply(xm, ym))
    r_den = k.backend.sqrt(tf.multiply(k.backend.sum(k.backend.square(xm)), k.backend.sum(k.backend.square(ym))))
    r = r_num / r_den

    r = k.backend.maximum(k.backend.minimum(r, 1.0), -1.0)

    return 1 - k.backend.square(r)


def accuracy_correlation_coefficient(y_true, y_pred):
    return (correlation_coefficient_loss(y_true, y_pred) * -1.0) + 1.0


def accuracy_correlation_coefficient_ignore_zero(y_true, y_pred):
    return (correlation_coefficient_loss(k.backend.cast(y_true > 0.0, "float32"),
                                         k.backend.cast(y_pred > 0.0, "float32")) * -1.0) + 1.0


# https://stackoverflow.com/questions/43855162/rmse-rmsle-loss-function-in-keras/43863854
def root_mean_squared_error(y_true, y_pred):
    return k.backend.sqrt(k.backend.mean(k.backend.square(y_pred - y_true)))


# https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def contractive_categorical_crossentropy(model, epsilon):
    # https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    def loss(y_pred, y_true):
        reconstruction_loss = k.losses.categorical_crossentropy(y_true, y_pred)

        w = k.backend.variable(value=model.get_layer("latent").get_weights()[0])  # N x N_hidden
        w = k.backend.transpose(w)  # N_hidden x N
        h = model.get_layer("latent").output
        dh = h * (1.0 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive_loss = \
            epsilon * k.backend.sum(k.backend.pow(dh, 2.0) * k.backend.sum(k.backend.pow(w, 2.0), axis=1), axis=1)

        return reconstruction_loss + contractive_loss
    return loss


# https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def kl_categorical_crossentropy(mu, sigma, beta):
    def loss(y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = k.losses.categorical_crossentropy(y_true, y_pred)

        # KL divergence loss
        kl_loss = 1.0 + sigma - k.backend.square(mu) - k.backend.exp(sigma)
        kl_loss = k.backend.sum(kl_loss, axis=-1)
        kl_loss = kl_loss * -0.5

        return reconstruction_loss + (beta * kl_loss)
    return loss


# https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def contractive_kl_categorical_crossentropy(mu, sigma, beta, model, epsilon):
    # https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    # https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/
    def loss(y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = k.losses.categorical_crossentropy(y_true, y_pred)

        w = k.backend.variable(value=model.get_layer('latent').get_weights()[0])  # N x N_hidden
        w = k.backend.transpose(w)  # N_hidden x N
        h = model.get_layer('latent').output
        dh = h * (1.0 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive_loss = \
            epsilon * k.backend.sum(k.backend.pow(dh, 2.0) * k.backend.sum(k.backend.pow(w, 2.0), axis=1), axis=1)

        # KL divergence loss
        kl_loss = 1.0 + sigma - k.backend.square(mu) - k.backend.exp(sigma)
        kl_loss = k.backend.sum(kl_loss, axis=-1)
        kl_loss = kl_loss * -0.5
        return reconstruction_loss + contractive_loss + (beta * kl_loss)
    return loss


# https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def contractive_correlation_coefficient_loss(model, epsilon):
    # https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    def loss(y_pred, y_true):
        reconstruction_loss = correlation_coefficient_loss(y_true, y_pred)

        w = k.backend.variable(value=model.get_layer("latent").get_weights()[0])  # N x N_hidden
        w = k.backend.transpose(w)  # N_hidden x N
        h = model.get_layer("latent").output
        dh = h * (1.0 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive_loss = \
            epsilon * k.backend.sum(k.backend.pow(dh, 2.0) * k.backend.sum(k.backend.pow(w, 2.0), axis=1), axis=1)

        return reconstruction_loss + contractive_loss
    return loss


# https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def kl_correlation_coefficient_loss(mu, sigma, beta):
    # https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/
    def loss(y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = correlation_coefficient_loss(y_true, y_pred)

        # KL divergence loss
        kl_loss = 1.0 + sigma - k.backend.square(mu) - k.backend.exp(sigma)
        kl_loss = k.backend.sum(kl_loss, axis=-1)
        kl_loss = kl_loss * -0.5

        return reconstruction_loss + (beta * kl_loss)
    return loss


# https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def contractive_kl_correlation_coefficient_loss(mu, sigma, beta, model, epsilon):
    # https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    # https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/
    def loss(y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = correlation_coefficient_loss(y_true, y_pred)

        w = k.backend.variable(value=model.get_layer('latent').get_weights()[0])  # N x N_hidden
        w = k.backend.transpose(w)  # N_hidden x N
        h = model.get_layer('latent').output
        dh = h * (1.0 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive_loss = \
            epsilon * k.backend.sum(k.backend.pow(dh, 2.0) * k.backend.sum(k.backend.pow(w, 2.0), axis=1), axis=1)

        # KL divergence loss
        kl_loss = 1.0 + sigma - k.backend.square(mu) - k.backend.exp(sigma)
        kl_loss = k.backend.sum(kl_loss, axis=-1)
        kl_loss = kl_loss * -0.5
        return reconstruction_loss + contractive_loss + (beta * kl_loss)
    return loss


# https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def contractive_log_cosh(model, epsilon):
    # https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    def loss(y_pred, y_true):
        reconstruction_loss = k.losses.log_cosh(y_true, y_pred)

        w = k.backend.variable(value=model.get_layer("latent").get_weights()[0])  # N x N_hidden
        w = k.backend.transpose(w)  # N_hidden x N
        h = model.get_layer("latent").output
        dh = h * (1.0 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive_loss = \
            epsilon * k.backend.sum(k.backend.pow(dh, 2.0) * k.backend.sum(k.backend.pow(w, 2.0), axis=1), axis=1)

        return reconstruction_loss + contractive_loss
    return loss


# https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def kl_log_cosh(mu, sigma, beta):
    # https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/
    def loss(y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = k.losses.log_cosh(y_true, y_pred)

        # KL divergence loss
        kl_loss = 1.0 + sigma - k.backend.square(mu) - k.backend.exp(sigma)
        kl_loss = k.backend.sum(kl_loss, axis=-1)
        kl_loss = kl_loss * -0.5

        return reconstruction_loss + (beta * kl_loss)
    return loss


# https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def contractive_kl_log_cosh(mu, sigma, beta, model, epsilon):
    # https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    # https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/
    def loss(y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = k.losses.log_cosh(y_true, y_pred)

        w = k.backend.variable(value=model.get_layer('latent').get_weights()[0])  # N x N_hidden
        w = k.backend.transpose(w)  # N_hidden x N
        h = model.get_layer('latent').output
        dh = h * (1.0 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive_loss = \
            epsilon * k.backend.sum(k.backend.pow(dh, 2.0) * k.backend.sum(k.backend.pow(w, 2.0), axis=1), axis=1)

        # KL divergence loss
        kl_loss = 1.0 + sigma - k.backend.square(mu) - k.backend.exp(sigma)
        kl_loss = k.backend.sum(kl_loss, axis=-1)
        kl_loss = kl_loss * -0.5
        return reconstruction_loss + contractive_loss + (beta * kl_loss)
    return loss
