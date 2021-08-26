# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import math
import numpy as np
import scipy.stats
import scipy.ndimage
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from tqdm import trange

import main
import parameters


if main.reproducible_bool:
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(main.seed_value)


def get_next_geometric_value(an, a0):
    print("get_next_geometric_value")

    n = math.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2, (math.ceil(n) - 1))

    return an


def data_upsample(data, data_type, new_resolution=None):
    print("data_upsample")

    for i in trange(len(data)):
        if data_type == "path":
            data_copy = np.load(data[i], mmap_mode="r+", allow_pickle=True)
        else:
            if data_type == "numpy":
                data_copy = data[i].copy()
            else:
                data_copy = None

        data_copy = np.squeeze(data_copy)

        if data_copy.ndim < 4:
            data_copy = np.expand_dims(data_copy, 0)

        if new_resolution is not None:
            dimension_x_upscale_factor = new_resolution[0] / data_copy.shape[1]
            dimension_y_upscale_factor = new_resolution[1] / data_copy.shape[2]
            dimension_z_upscale_factor = new_resolution[2] / data_copy.shape[3]
        else:
            data_copy_shape = list(data_copy.shape)

            for j in trange(len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            geometric_sequence_a0 = 2

            dimension_x_upscale_factor = \
                get_next_geometric_value(data_copy_shape[1], geometric_sequence_a0) / data_copy.shape[1]
            dimension_y_upscale_factor = \
                get_next_geometric_value(data_copy_shape[2], geometric_sequence_a0) / data_copy.shape[2]
            dimension_z_upscale_factor = \
                get_next_geometric_value(data_copy_shape[3], geometric_sequence_a0) / data_copy.shape[3]

        if not np.isclose(dimension_x_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_y_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_z_upscale_factor, 1.0, rtol=0.0, atol=1e-05):
            data_copy = scipy.ndimage.zoom(data_copy, (1, dimension_x_upscale_factor, dimension_y_upscale_factor,
                                                       dimension_z_upscale_factor), order=2, mode="mirror",
                                           prefilter=True)

        data_copy = np.expand_dims(data_copy, -1)

        if data_type == "path":
            np.save(data[i], data_copy, allow_pickle=True, fix_imports=False)
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data


# https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/preprocessing/_data.py#L2938
def yeo_johnson_inverse_transform(power_transformer, x):
    def _yeo_johnson_inverse_transform(_x, _lmbda):
        _x_inv = np.zeros_like(_x)
        pos = (_x >= 0)

        if np.count_nonzero(pos) > 0:
            # when x >= 0
            if abs(_lmbda) < np.spacing(1.0):
                _x_inv[pos] = np.exp(_x[pos]) - 1.0
            else:  # lmbda != 0
                _x_inv[pos] = np.power((_x[pos] * _lmbda) + 1.0, 1.0 / _lmbda) - 1.0

            _x_inv[pos] = np.nan_to_num(_x_inv[pos], copy=False, nan=float(np.nanmax(_x_inv[pos])))

        if np.count_nonzero(~pos) > 0:
            # when x < 0
            if abs(_lmbda - 2.0) > np.spacing(1.0):
                _x_inv[~pos] = 1.0 - np.power((-(2.0 - _lmbda) * _x[~pos]) + 1.0, 1.0 / (2.0 - _lmbda))
            else:  # lmbda == 2
                _x_inv[~pos] = 1.0 - np.exp(-_x[~pos])

            _x_inv[~pos] = np.nan_to_num(_x_inv[~pos], copy=False, nan=float(np.nanmin(_x_inv[~pos])))

        return _x_inv

    x_inv = x.copy()

    for i, lmbda in enumerate(power_transformer.lambdas_):
        x_inv[:, i] = _yeo_johnson_inverse_transform(x_inv[:, i], lmbda)

    return x_inv


def data_preprocessing(data, data_type, preprocessing_steps=None):
    print("data_preprocessing")

    if preprocessing_steps is None:
        preprocessing_steps = []

        for _ in trange(len(data)):
            preprocessing_steps.append(None)

    for i in trange(len(data)):
        if data_type == "path":
            data_copy = np.load(data[i], mmap_mode="r+", allow_pickle=True)
        else:
            if data_type == "numpy":
                data_copy = data[i].copy()
            else:
                data_copy = None

        data_copy_shape = data_copy.shape
        data_copy = data_copy.reshape(-1, 1)

        if preprocessing_steps[i] is None:
            current_preprocessing_steps = [StandardScaler(copy=False)]
            data_copy = current_preprocessing_steps[-1].fit_transform(data_copy)

            current_preprocessing_steps.append(PowerTransformer(standardize=False, copy=False))
            data_copy = current_preprocessing_steps[-1].fit_transform(data_copy)

            current_preprocessing_steps.append(StandardScaler(copy=False))
            data_copy = current_preprocessing_steps[-1].fit_transform(data_copy)

            preprocessing_steps[i] = current_preprocessing_steps
        else:
            data_copy = preprocessing_steps[i][2].inverse_transform(data_copy)
            data_copy = yeo_johnson_inverse_transform(preprocessing_steps[i][1], data_copy)
            data_copy = preprocessing_steps[i][0].inverse_transform(data_copy)

            data_copy_background = scipy.stats.mode(data_copy, axis=None, nan_policy="omit")[0][0]
            data_copy[data_copy < data_copy_background] = data_copy_background
            data_copy = data_copy - data_copy_background

        data_copy = data_copy.reshape(data_copy_shape)

        if data_type == "path":
            np.save(data[i], data_copy, allow_pickle=True, fix_imports=False)
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data, preprocessing_steps


def redistribute(data, data_type, robust_bool=False, new_distribution=None, new_distribution_type=None):
    print("redistribute")

    for i in trange(len(data)):
        if data_type == "path":
            data_copy = np.load(data[i], mmap_mode="r+", allow_pickle=True)
        else:
            if data_type == "numpy":
                data_copy = data[i].copy()
            else:
                data_copy = None

        data_copy_shape = data_copy.shape
        data_copy = data_copy.reshape(-1, 1)

        if robust_bool:
            data_copy = RobustScaler(copy=False).fit_transform(data_copy)
        else:
            data_copy = StandardScaler(copy=False).fit_transform(data_copy)

        if new_distribution is not None:
            if new_distribution_type == "path":
                new_distribution_copy = np.load(new_distribution[i], mmap_mode="r+", allow_pickle=True)
            else:
                if new_distribution_type == "numpy":
                    new_distribution_copy = new_distribution[i].copy()
                else:
                    new_distribution_copy = None

            new_distribution_copy = new_distribution_copy.reshape(-1, 1)

            if robust_bool:
                standard_scaler = RobustScaler(copy=False)
            else:
                standard_scaler = StandardScaler(copy=False)

            standard_scaler.fit(new_distribution_copy)
            data_copy = standard_scaler.inverse_transform(data_copy)

        data_copy = data_copy.reshape(data_copy_shape)

        if data_type == "path":
            np.save(data[i], data_copy, allow_pickle=True, fix_imports=False)
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data
