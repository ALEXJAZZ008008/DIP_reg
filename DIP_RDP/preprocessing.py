# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import math
import numpy as np
import scipy.stats
import scipy.ndimage
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler, \
    Normalizer
from sklearn.decomposition import PCA
import skimage.transform
import skimage.measure
import elasticdeform

import parameters

rng = np.random.default_rng()


def get_next_geometric_value(an, a0):
    print("get_next_geometric_value")

    n = math.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2, (math.ceil(n) - 1))

    return an


def data_upsample(data, new_resolution=None):
    print("pad_data_resample")

    geometric_sequence_a0 = 3

    for i in range(len(data)):
        data_copy = np.load(data[i], allow_pickle=True)

        data_copy = np.squeeze(data_copy)

        if new_resolution is not None:
            dimension_x_upscale_factor = new_resolution[0] / data_copy.shape[0]
            dimension_y_upscale_factor = new_resolution[1] / data_copy.shape[1]
            dimension_z_upscale_factor = new_resolution[2] / data_copy.shape[2]
        else:
            dimension_x_upscale_factor = \
                get_next_geometric_value(data_copy.shape[0], geometric_sequence_a0) / data_copy.shape[0]
            dimension_y_upscale_factor = \
                get_next_geometric_value(data_copy.shape[1], geometric_sequence_a0) / data_copy.shape[1]
            dimension_z_upscale_factor = \
                get_next_geometric_value(data_copy.shape[2], geometric_sequence_a0) / data_copy.shape[2]

        if not np.isclose(dimension_x_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_y_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_z_upscale_factor, 1.0, rtol=0.0, atol=1e-05):
            data_copy = scipy.ndimage.zoom(data_copy, (dimension_x_upscale_factor, dimension_y_upscale_factor,
                                                       dimension_z_upscale_factor), order=3, mode="mirror",
                                           prefilter=True)

        data_copy = np.expand_dims(data_copy, -1)

        np.save(data[i], data_copy)

    return data


def data_downsample(data, resample_even_bool, downsample_factor):
    print("data_downsample")

    for i in range(len(data)):
        data_copy = np.load(data[i], allow_pickle=True)

        data_copy = np.squeeze(data_copy)

        if resample_even_bool:
            downsampled_data_copy = []

            for j in range(len(data_copy)):
                downsampled_data_copy.append(skimage.transform.pyramid_reduce(data_copy[j], downscale=downsample_factor,
                                                                              order=3, mode="mirror",
                                                                              multichannel=False, preserve_range=True))

            data_copy = np.asfarray(downsampled_data_copy)
        else:
            downsampled_data_copy = []

            for j in range(len(data_copy)):
                downsampled_data_copy.append(
                    skimage.measure.block_reduce(data_copy[j],
                                                 (downsample_factor, downsample_factor, downsample_factor),
                                                 func=np.mean))

            data_copy = np.asfarray(downsampled_data_copy)

        data_copy = np.expand_dims(data_copy, -1)

        np.save(data[i], data_copy)

    return data


def data_preprocessing(data, robust_bool):
    print("data_preprocessing")

    power_transformer_bool = True
    quantile_transformer_bool = False
    robust_scaler_bool = robust_bool
    pca_bool = False
    normaliser_bool = False

    for i in range(len(data)):
        data_copy = np.load(data[i], allow_pickle=True)

        data_copy_shape = data_copy.shape
        data_copy = data_copy.reshape(-1, 1)

        data_copy_mode_feature_index = \
            np.squeeze(np.isclose(data_copy, scipy.stats.mode(data_copy, axis=None, nan_policy="omit")[0][0], rtol=0.0,
                                  atol=1e-05))

        if power_transformer_bool:
            normal_transformer_prescaler = \
                Pipeline([("scaler", StandardScaler(copy=False)),
                          ("min_max_scaler", MinMaxScaler(feature_range=(0.5, 1.5), copy=False))])
            normal_transformer_prescaler.fit(data_copy[~ data_copy_mode_feature_index, :])

            data_copy = normal_transformer_prescaler._transform(data_copy)

            power_transformer = PowerTransformer(method="box-cox", standardize=False, copy=False)
            power_transformer.fit(data_copy[~ data_copy_mode_feature_index, :])

            data_copy = power_transformer.transform(data_copy)
        else:
            if quantile_transformer_bool:
                normal_transformer_prescaler = \
                    Pipeline([("scaler", StandardScaler(copy=False)),
                              ("min_max_scaler", MinMaxScaler(feature_range=(0.5, 1.5), copy=False)),
                              ("power_transformer", PowerTransformer(method="box-cox", standardize=True, copy=False))])
                normal_transformer_prescaler.fit(data_copy[~ data_copy_mode_feature_index, :])

                data = normal_transformer_prescaler.transform(data_copy)

                quantile_transformer = QuantileTransformer(n_quantiles=data_copy.reshape(data_copy.shape[0],
                                                                                         -1).shape[0],
                                                           subsample=data_copy.reshape(data_copy.shape[0],
                                                                                       -1).shape[0],
                                                           output_distribution="normal",
                                                           random_state=np.random.RandomState(), copy=False)
                data_copy = quantile_transformer.fit(data_copy[~ data_copy_mode_feature_index, :])

                data_copy = quantile_transformer.transform(data_copy)

        if not pca_bool and robust_scaler_bool:
            scaler = RobustScaler(copy=False, unit_variance=True)
        else:
            scaler = StandardScaler(copy=False)

        scaler.fit(data_copy[~ data_copy_mode_feature_index, :])

        data_copy = scaler.transform(data_copy)

        if pca_bool:
            pca = PCA(n_components="mle", copy=False, whiten=True, svd_solver="full")
            pca.fit(data_copy[~ data_copy_mode_feature_index, :])

            data_copy = pca.transform(data_copy)
            data_copy = pca.inverse_transform(data_copy)

        if normaliser_bool:
            normalizer = Normalizer(copy=False)
            normalizer.fit(data_copy[~ data_copy_mode_feature_index, :])

            data_copy = normalizer.transform(data_copy)

        data_copy = data_copy.reshape(data_copy_shape)

        np.save(data[i], data_copy)

    return data


def intermediate_data_preprocessing(data, robust_bool):
    print("intermediate_data_preprocessing")

    power_transformer_bool = True
    quantile_transformer_bool = False
    robust_scaler_bool = robust_bool
    normaliser_bool = False

    data_shape = data.shape
    data = data.reshape(-1, 1)

    data_mode_feature_index = \
        np.squeeze(np.isclose(data, scipy.stats.mode(data, axis=None, nan_policy="omit")[0][0], rtol=0.0,
                              atol=1e-05))

    if power_transformer_bool:
        normal_transformer_prescaler = \
            Pipeline([("scaler", StandardScaler(copy=False)),
                      ("min_max_scaler", MinMaxScaler(feature_range=(0.5, 1.5), copy=False))])
        normal_transformer_prescaler.fit(data[~ data_mode_feature_index, :])

        data = normal_transformer_prescaler.transform(data)

        power_transformer = PowerTransformer(method="box-cox", standardize=False, copy=False)
        power_transformer.fit(data[~ data_mode_feature_index, :])

        data = power_transformer.transform(data)
    else:
        if quantile_transformer_bool:
            normal_transformer_prescaler = \
                Pipeline([("scaler", StandardScaler(copy=False)),
                          ("min_max_scaler", MinMaxScaler(feature_range=(0.5, 1.5), copy=False)),
                          ("power_transformer", PowerTransformer(method="box-cox", standardize=True, copy=False))])
            normal_transformer_prescaler.fit(data[~ data_mode_feature_index, :])

            data = normal_transformer_prescaler.transform(data)

            quantile_transformer = \
                QuantileTransformer(n_quantiles=data.reshape(data.shape[0], -1).shape[0],
                                    subsample=data.reshape(data.shape[0], -1).shape[0],
                                    output_distribution="normal",
                                    random_state=np.random.RandomState(), copy=False)
            quantile_transformer.fit(data[~ data_mode_feature_index, :])

            data = quantile_transformer.transform(data)

    if robust_scaler_bool:
        scaler = RobustScaler(copy=False, unit_variance=True)
    else:
        scaler = StandardScaler(copy=False)

    scaler.fit(data[~ data_mode_feature_index, :])

    data = scaler.transform(data)

    if normaliser_bool:
        normalizer = Normalizer(copy=False)
        normalizer.fit(data[~ data_mode_feature_index, :])

        data = normalizer.transform(data)

    data[~ data_mode_feature_index, :] = \
        scipy.stats.mode(data[~ data_mode_feature_index, :], axis=1)[0]

    data = data.reshape(data_shape)

    return data


def elastic_deform_data(x_data, y_data):
    print("elastic_deform_data")

    elastic_sigma = parameters.elastic_sigma

    if elastic_sigma > 0.0:
        elastic_spacing = 5
        points = [int(x_data.shape[0] / elastic_spacing),
                  int(x_data.shape[1] / elastic_spacing),
                  int(x_data.shape[2] / elastic_spacing)]

        # pycharm gives incorrect type warning, hence turning it off
        # warning possible issues in the future
        if y_data is not None:
            # noinspection PyTypeChecker
            [x_data, y_data] = \
                elasticdeform.deform_random_grid([x_data, y_data], sigma=elastic_sigma, points=points, order=3,
                                                 mode="mirror", prefilter=True, axis=[(0, 1, 2), (0, 1, 2)])
        else:
            # noinspection PyTypeChecker
            x_data = elasticdeform.deform_random_grid(x_data, sigma=elastic_sigma, points=points, order=3,
                                                      mode="mirror", prefilter=True, axis=[(0, 1, 2)])

    return x_data, y_data


def get_identity_transformation_matrix():
    print("get_identity_transformation_matrix")

    identity_matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    return identity_matrix


# https://stackoverflow.com/questions/14926798/python-transformation-matrix
def trig(angle):
    print("trig")

    r = math.radians(angle)

    return math.cos(r), math.sin(r)


# https://stackoverflow.com/questions/14926798/python-transformation-matrix
def get_srt_transformation_matrix(origin_translation, scale, rotation, translation=None):
    print("get_srt_transformation_matrix")

    odx, ody, odz = origin_translation

    s = scale[0]

    if s <= 0:
        s = 1.0

    rcx, rsx = trig(rotation[0])
    rcy, rsy = trig(rotation[1])
    rcz, rsz = trig(rotation[2])

    if translation is not None:
        dx = translation[0]
        dy = translation[1]
        dz = translation[1]
    else:
        dx = None
        dy = None
        dz = None

    origin_translate_matrix = np.array([[1, 0, 0, -odx],
                                        [0, 1, 0, -ody],
                                        [0, 0, 1, -odz],
                                        [0, 0, 0, 1]])

    scale_matrix = np.array([[s, 0, 0, 0],
                             [0, s, 0, 0],
                             [0, 0, s, 0],
                             [0, 0, 0, 1]])

    x_rotate_matrix = np.array([[1, 0, 0, 0],
                                [0, rcx, -rsx, 0],
                                [0, rsx, rcx, 0],
                                [0, 0, 0, 1]])

    y_rotate_matrix = np.array([[rcy, 0, rsy, 0],
                                [0, 1, 0, 0],
                                [-rsy, 0, rcy, 0],
                                [0, 0, 0, 1]])

    z_rotate_matrix = np.array([[rcz, -rsz, 0, 0],
                                [rsz, rcz, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    if translation is not None:
        translate_matrix = np.array([[1, 0, 0, dx],
                                     [0, 1, 0, dy],
                                     [0, 0, 1, dz],
                                     [0, 0, 0, 1]])
    else:
        translate_matrix = get_identity_transformation_matrix()

    inverse_origin_translate_matrix = np.array([[1, 0, 0, odx],
                                                [0, 1, 0, ody],
                                                [0, 0, 1, odz],
                                                [0, 0, 0, 1]])

    output_transform = np.dot(scale_matrix, origin_translate_matrix)
    output_transform = np.dot(x_rotate_matrix, output_transform)
    output_transform = np.dot(y_rotate_matrix, output_transform)
    output_transform = np.dot(z_rotate_matrix, output_transform)
    output_transform = np.dot(translate_matrix, output_transform)
    output_transform = np.dot(inverse_origin_translate_matrix, output_transform)

    return output_transform


def get_truncated_normal_value(low=0.0, upp=10.0):
    print("get_truncated_normal_value")

    return (scipy.stats.truncnorm((0.0 - 0.5) / 0.5, (1.0 - 0.5) / 0.5, loc=0.5, scale=0.5).rvs() * (upp - low)) - \
           (low * -1)


def transform_data(x_data, y_data):
    print("transform_data")

    scale_bool = True
    rotate_bool = True
    translate_bool = True

    origin_translation = [x_data.shape[0] / 2.0, x_data.shape[1] / 2.0, x_data.shape[2] / 2.0]

    if scale_bool:
        min_scale = 0.9
        max_scale = 1.2
    else:
        min_scale = 1.0
        max_scale = 1.0

    if rotate_bool:
        min_rotation_degrees = -5.0
        max_rotation_degrees = 5.0
    else:
        min_rotation_degrees = 0.0
        max_rotation_degrees = 0.0

    if translate_bool:
        max_translate = 16
    else:
        max_translate = 0

    srt_transformation_matrix = \
        get_srt_transformation_matrix(origin_translation,
                                      [get_truncated_normal_value(min_scale, max_scale)],
                                      [get_truncated_normal_value(min_rotation_degrees, max_rotation_degrees),
                                       get_truncated_normal_value(min_rotation_degrees, max_rotation_degrees),
                                       get_truncated_normal_value(min_rotation_degrees, max_rotation_degrees)],
                                      [get_truncated_normal_value(0, max_translate),
                                       get_truncated_normal_value(0, max_translate),
                                       get_truncated_normal_value(0, max_translate)])

    for i in range(len(x_data)):
        x_data[i] = \
            np.expand_dims(scipy.ndimage.affine_transform(
                x_data[i].reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2])),
                np.linalg.inv(srt_transformation_matrix), mode="mirror", prefilter=True), -1)

        if y_data is not None:
            y_data[i] = \
                np.expand_dims(scipy.ndimage.affine_transform(
                    y_data[i].reshape((y_data.shape[0], y_data.shape[1], x_data.shape[2])),
                    np.linalg.inv(srt_transformation_matrix), mode="mirror", prefilter=True), -1)

    return x_data, y_data


def get_transformed_data(x_data, y_data, robust_bool, output_path):
    print("get_transformed_data")

    transform_bool = parameters.transform_bool
    repreprocess_bool = False

    output_x_data = x_data.copy()

    if y_data is not None:
        output_y_data = y_data.copy()
    else:
        output_y_data = None

    if transform_bool:
        if not os.path.exists("{0}_transformed".format(output_path)):
            os.makedirs("{0}_transformed".format(output_path), mode=0o770)

        if not os.path.exists("{0}_transformed/x".format(output_path)):
            os.makedirs("{0}_transformed/x".format(output_path), mode=0o770)

        if not os.path.exists("{0}_transformed/y".format(output_path)):
            os.makedirs("{0}_transformed/y".format(output_path), mode=0o770)

        for i in range(len(output_x_data)):
            x_data_copy = np.load(output_x_data[i], allow_pickle=True)

            x_data_copy_mean = np.mean(x_data_copy)
            x_data_copy_std = np.std(x_data_copy)

            if output_y_data is not None:
                y_data_copy = np.load(output_y_data[i], allow_pickle=True)

                y_data_copy_mean = np.mean(y_data_copy)
                y_data_copy_std = np.std(y_data_copy)
            else:
                y_data_copy = None

                y_data_copy_mean = None
                y_data_copy_std = None

            x_data_copy, y_data_copy = elastic_deform_data(x_data_copy, y_data_copy)

            x_data_copy, y_data_copy = transform_data(x_data_copy, y_data_copy)

            if repreprocess_bool:
                x_data_copy = intermediate_data_preprocessing(x_data_copy, robust_bool)

                if output_y_data is not None:
                    y_data_copy = intermediate_data_preprocessing(y_data_copy, robust_bool)
            else:
                x_data_copy = \
                    (((x_data_copy - np.mean(x_data_copy)) / np.std(x_data_copy)) * x_data_copy_std) + x_data_copy_mean

                if output_y_data is not None:
                    y_data_copy = \
                        (((y_data_copy - np.mean(y_data_copy)) / np.std(y_data_copy))
                         * y_data_copy_std) + y_data_copy_mean

            output_x_data[i] = "{0}_transformed/x/{1}.npy".format(output_path, str(i))

            np.save(output_x_data[i], x_data_copy)

            if output_y_data is not None:
                output_y_data[i] = "{0}_transformed/y/{1}.npy".format(output_path, str(i))

                np.save(output_y_data[i], y_data_copy)

    return output_x_data, output_y_data


def get_noisy_data(data, robust_bool, output_path):
    print("get_noisy_data")

    noise_scale = parameters.noise_scale
    background_scale = parameters.background_scale
    repreprocess_bool = False

    output_data = data.copy()

    if noise_scale > 0.0:
        if not os.path.exists("{0}_noisy".format(output_path)):
            os.makedirs("{0}_noisy".format(output_path), mode=0o770)

        for i in range(len(output_data)):
            data_copy = np.load(output_data[i], allow_pickle=True)

            data_copy_mean = np.mean(data_copy)
            data_copy_std = np.std(data_copy)

            data_copy = data_copy - np.min(data_copy)

            if background_scale > 0.0:
                data_copy = data_copy + (np.mean(data_copy) / background_scale)

            data_copy = data_copy / np.std(data_copy)

            data_copy = rng.poisson(data_copy * (1.0 / noise_scale))

            if repreprocess_bool:
                data_copy = intermediate_data_preprocessing(data_copy, robust_bool)
            else:
                data_copy = (((data_copy - np.mean(data_copy)) / np.std(data_copy)) * data_copy_std) + data_copy_mean

            output_data[i] = "{0}_noisy/{1}.npy".format(output_path, str(i))

            np.save(output_data[i], data_copy)

    return output_data
