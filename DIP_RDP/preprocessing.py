# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import math
import numpy as np
import scipy.stats
import scipy.ndimage
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler


import main

if main.reproducible_bool:
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(main.seed_value)


import parameters


def get_next_geometric_value(an, a0):
    print("get_next_geometric_value")

    n = math.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2, (math.ceil(n) - 1))

    return an


def get_previous_geometric_value(an, a0):
    print("get_previous_geometric_value")

    n = math.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2, (math.floor(n) - 1))

    return an


def data_upsample(data, data_type, new_resolution=None):
    print("data_upsample")

    for i in range(len(data)):
        if data_type == "path":
            data_copy = np.load(data[i])
        else:
            if data_type == "numpy":
                data_copy = data[i].copy()
            else:
                data_copy = None

        data_copy = np.squeeze(data_copy)

        if data_copy.ndim < 4:
            data_copy = np.expand_dims(data_copy, 0)

        if new_resolution is not None:
            if (new_resolution[0] % 2) != (data_copy.shape[1] % 2):
                dimension_x_upscale_factor = (data_copy.shape[1] + 1.0) / data_copy.shape[1]
            else:
                dimension_x_upscale_factor = 1.0

            if (new_resolution[1] % 2) != (data_copy.shape[2] % 2):
                dimension_y_upscale_factor = (data_copy.shape[2] + 1.0) / data_copy.shape[2]
            else:
                dimension_y_upscale_factor = 1.0

            if (new_resolution[2] % 2) != (data_copy.shape[3] % 2):
                dimension_z_upscale_factor = (data_copy.shape[3] + 1.0) / data_copy.shape[3]
            else:
                dimension_z_upscale_factor = 1.0
        else:
            data_copy_shape = list(data_copy.shape)

            for j in range(1, len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            data_copy_shape = [1, get_next_geometric_value(data_copy_shape[1], parameters.data_resample_power_of),
                               get_next_geometric_value(data_copy_shape[2], parameters.data_resample_power_of),
                               get_next_geometric_value(data_copy_shape[3], parameters.data_resample_power_of)]

            if (data_copy_shape[1] % 2) != (data_copy.shape[1] % 2):
                dimension_x_upscale_factor = (data_copy.shape[1] + 1.0) / data_copy.shape[1]
            else:
                dimension_x_upscale_factor = 1.0

            if (data_copy_shape[2] % 2) != (data_copy.shape[2] % 2):
                dimension_y_upscale_factor = (data_copy.shape[2] + 1.0) / data_copy.shape[2]
            else:
                dimension_y_upscale_factor = 1.0

            if (data_copy_shape[3] % 2) != (data_copy.shape[3] % 2):
                dimension_z_upscale_factor = (data_copy.shape[3] + 1.0) / data_copy.shape[3]
            else:
                dimension_z_upscale_factor = 1.0

        if not np.isclose(dimension_x_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_y_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_z_upscale_factor, 1.0, rtol=0.0, atol=1e-05):
            data_copy = scipy.ndimage.zoom(data_copy, (1, dimension_x_upscale_factor, dimension_y_upscale_factor,
                                                       dimension_z_upscale_factor), order=1, mode="mirror",
                                           prefilter=True)

        if new_resolution is not None:
            dimension_x_pad_factor = int((new_resolution[0] - data_copy.shape[1]) / 2.0)
            dimension_y_pad_factor = int((new_resolution[1] - data_copy.shape[2]) / 2.0)
            dimension_z_pad_factor = int((new_resolution[2] - data_copy.shape[3]) / 2.0)
        else:
            data_copy_shape = list(data_copy.shape)

            for j in range(1, len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            dimension_x_pad_factor = \
                int((get_next_geometric_value(data_copy_shape[1], parameters.data_resample_power_of) -
                    data_copy.shape[1]) / 2.0)
            dimension_y_pad_factor = \
                int((get_next_geometric_value(data_copy_shape[2], parameters.data_resample_power_of) -
                     data_copy.shape[2]) / 2.0)
            dimension_z_pad_factor = \
                int((get_next_geometric_value(data_copy_shape[3], parameters.data_resample_power_of) -
                     data_copy.shape[3]) / 2.0)

        if dimension_x_pad_factor != 0 or dimension_y_pad_factor != 0 or dimension_z_pad_factor != 0:
            data_copy = np.pad(data_copy, ((0, 0), (dimension_x_pad_factor, dimension_x_pad_factor),
                                           (dimension_y_pad_factor, dimension_y_pad_factor),
                                           (dimension_z_pad_factor, dimension_z_pad_factor)), mode="reflect")

        data_copy = np.expand_dims(data_copy, -1)

        if data_type == "path":
            np.save(data[i], data_copy)
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data


def data_downsampling(data, data_type, new_resolution=None):
    print("data_downsampling")

    for i in range(len(data)):
        if data_type == "path":
            data_copy = np.load(data[i])
        else:
            if data_type == "numpy":
                data_copy = data[i].copy()
            else:
                data_copy = None

        data_copy = np.squeeze(data_copy)

        if data_copy.ndim < 4:
            data_copy = np.expand_dims(data_copy, 0)

        if new_resolution is not None:
            dimension_x_crop_factor = int(np.floor((data_copy.shape[1] - new_resolution[0]) / 2.0))
            dimension_y_crop_factor = int(np.floor((data_copy.shape[2] - new_resolution[1]) / 2.0))
            dimension_z_crop_factor = int(np.floor((data_copy.shape[3] - new_resolution[2]) / 2.0))
        else:
            data_copy_shape = list(data_copy.shape)

            for j in range(1, len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            dimension_x_crop_factor = \
                int((data_copy.shape[1] -
                     get_previous_geometric_value(data_copy_shape[1], parameters.data_resample_power_of)) / 2.0)
            dimension_y_crop_factor = \
                int((data_copy.shape[2] -
                     get_previous_geometric_value(data_copy_shape[2], parameters.data_resample_power_of)) / 2.0)
            dimension_z_crop_factor = \
                int((data_copy.shape[3] -
                     get_previous_geometric_value(data_copy_shape[3], parameters.data_resample_power_of)) / 2.0)

        if dimension_x_crop_factor != 0 or dimension_y_crop_factor != 0 or dimension_z_crop_factor != 0:
            data_copy = data_copy[:, dimension_x_crop_factor or None:-dimension_x_crop_factor or None,
                        dimension_y_crop_factor or None:-dimension_y_crop_factor or None,
                        dimension_z_crop_factor or None:-dimension_z_crop_factor or None]

        if new_resolution is not None:
            dimension_x_downscale_factor = new_resolution[0] / data_copy.shape[1]
            dimension_y_downscale_factor = new_resolution[1] / data_copy.shape[2]
            dimension_z_downscale_factor = new_resolution[2] / data_copy.shape[3]
        else:
            data_copy_shape = list(data_copy.shape)

            for j in range(1, len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            data_copy_shape = [1, get_previous_geometric_value(data_copy_shape[1], parameters.data_resample_power_of),
                               get_previous_geometric_value(data_copy_shape[2], parameters.data_resample_power_of),
                               get_previous_geometric_value(data_copy_shape[3], parameters.data_resample_power_of)]

            dimension_x_downscale_factor = data_copy_shape[1] / data_copy.shape[1]
            dimension_y_downscale_factor = data_copy_shape[2] / data_copy.shape[2]
            dimension_z_downscale_factor = data_copy_shape[3] / data_copy.shape[3]

        if not np.isclose(dimension_x_downscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_y_downscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_z_downscale_factor, 1.0, rtol=0.0, atol=1e-05):
            data_copy = scipy.ndimage.zoom(data_copy, (1, dimension_x_downscale_factor, dimension_y_downscale_factor,
                                                       dimension_z_downscale_factor), order=1, mode="mirror",
                                           prefilter=True)

        data_copy = np.expand_dims(data_copy, -1)

        if data_type == "path":
            np.save(data[i], data_copy)
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

            x_inv_pos_max = float(np.nanmax(_x_inv[pos]))
            _x_inv[pos] = np.nan_to_num(_x_inv[pos], copy=False, nan=x_inv_pos_max, posinf=x_inv_pos_max,
                                        neginf=float(np.nanmin(_x_inv[pos])))

        if np.count_nonzero(~pos) > 0:
            # when x < 0
            if abs(_lmbda - 2.0) > np.spacing(1.0):
                _x_inv[~pos] = 1.0 - np.power((-(2.0 - _lmbda) * _x[~pos]) + 1.0, 1.0 / (2.0 - _lmbda))
            else:  # lmbda == 2
                _x_inv[~pos] = 1.0 - np.exp(-_x[~pos])

            x_inv_not_pos_min = float(np.nanmin(_x_inv[~pos]))
            _x_inv[~pos] = np.nan_to_num(_x_inv[~pos], copy=False, nan=x_inv_not_pos_min,
                                         posinf=float(np.nanmax(_x_inv[~pos])), neginf=x_inv_not_pos_min)

        return _x_inv

    x_inv = x.copy()

    for i, lmbda in enumerate(power_transformer.lambdas_):
        x_inv[:, i] = _yeo_johnson_inverse_transform(x_inv[:, i], lmbda)

    return x_inv


def data_preprocessing(data, data_type, preprocessing_steps=None):
    print("data_preprocessing")

    if preprocessing_steps is None:
        preprocessing_steps = []

        for _ in range(len(data)):
            preprocessing_steps.append(None)

    for i in range(len(data)):
        if data_type == "path":
            data_copy = np.load(data[i])
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
            np.save(data[i], data_copy)
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data, preprocessing_steps


def redistribute(data, data_type, robust_bool=False, new_distribution=None, new_distribution_type=None):
    print("redistribute")

    for i in range(len(data)):
        if data_type == "path":
            data_copy = np.load(data[i])
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
                new_distribution_copy = np.load(new_distribution[i])
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
            np.save(data[i], data_copy)
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data
