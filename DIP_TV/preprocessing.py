# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


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

rng = np.random.default_rng()


def get_next_geometric_value(an, a0):
    print("get_next_geometric_value")

    n = math.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2, (math.ceil(n) - 1))

    return an


def data_upsample(data, new_resolution=None):
    print("pad_data_resample")

    geometric_sequence_a0 = 2

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

    power_transformer_bool = False
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
