# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import math
import numpy as np
import scipy.stats
import scipy.ndimage
from sklearn.preprocessing import StandardScaler
from tqdm import trange

rng = np.random.default_rng()


def get_next_geometric_value(an, a0):
    print("get_next_geometric_value")

    n = math.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2, (math.ceil(n) - 1))

    return an


def data_upsample(data, data_type, new_resolution=None):
    print("data_upsample")

    geometric_sequence_a0 = 2

    for i in trange(len(data)):
        if data_type == "path":
            data_copy = np.load(data[i], allow_pickle=True)
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
            dimension_x_upscale_factor = \
                get_next_geometric_value(data_copy.shape[1], geometric_sequence_a0) / data_copy.shape[1]
            dimension_y_upscale_factor = \
                get_next_geometric_value(data_copy.shape[2], geometric_sequence_a0) / data_copy.shape[2]
            dimension_z_upscale_factor = \
                get_next_geometric_value(data_copy.shape[3], geometric_sequence_a0) / data_copy.shape[3]

        if not np.isclose(dimension_x_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_y_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_z_upscale_factor, 1.0, rtol=0.0, atol=1e-05):
            data_copy = scipy.ndimage.zoom(data_copy, (1, dimension_x_upscale_factor, dimension_y_upscale_factor,
                                                       dimension_z_upscale_factor), order=3, mode="mirror",
                                           prefilter=True)

        data_copy = np.expand_dims(data_copy, -1)

        if data_type == "path":
            np.save(data[i], data_copy)
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data


def data_preprocessing(data, data_type, scalers=None):
    print("data_preprocessing")

    if scalers is None:
        scalers = []

    for i in trange(len(data)):
        if data_type == "path":
            data_copy = np.load(data[i], allow_pickle=True)
        else:
            if data_type == "numpy":
                data_copy = data[i].copy()
            else:
                data_copy = None

        data_copy_shape = data_copy.shape
        data_copy = data_copy.reshape(-1, 1)

        if len(scalers) < len(data):
            scaler = StandardScaler()
            scaler.fit(data_copy)

            scalers.append(scaler)

            data_copy = scaler.transform(data_copy)
        else:
            data_copy = scalers[i].inverse_transform(data_copy)

        data_copy = data_copy.reshape(data_copy_shape)

        if data_type == "path":
            np.save(data[i], data_copy)
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data, scalers
