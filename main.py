# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import re
import shutil
import hurry.filesize
import random
import numpy as np
import scipy.constants
import scipy.stats
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
import pickle
import nibabel

import parameters
import transcript
import preprocessing
import architecture


random.seed()


float_sixteen_bool = True  # set the network to use float16 data
cpu_bool = False  # if using CPU, set to true: disables mixed precision computation
cluster_bool = False  # if using cluster, set to true: disables additional downsampling

# mixed precision float16 computation allows the network to use both float16 and float32 where necessary,
# this improves performance on the GPU.
if float_sixteen_bool and not cpu_bool:
    policy = k.mixed_precision.Policy("mixed_float16")
    k.mixed_precision.set_global_policy(policy)
else:
    policy = k.mixed_precision.Policy(tf.dtypes.float32.name)
    k.mixed_precision.set_global_policy(policy)


train_bool = parameters.train_bool

debug_bool = True
save_preprocessing_bool = True
plot_bool = False
save_plot_bool = True
data_path = "{0}/AI-Project_data/xcat/".format(os.path.dirname(os.getcwd()))
output_path = "{0}/output".format(os.getcwd())

autoencoder_bool = parameters.autoencoder_bool
classifier_bool = parameters.classifier_bool

# suggested values
# follow for both from left to right for pseudo simulated annealing
# follow in inverse directions for traditional
# 1e-01 * (n * 0.1):    1e-01,  1e-02,  1e-03,  1e-04,  1e-05,  1e-06   etc
initial_learning_rate = parameters.initial_learning_rate
# if using 4d data then each sample can be considered a batch for time distributed layers
# rounded down to nearest power of two
# 1 * (n * (1 / 0.1)):  1,      8,      64,     512,    8192,   65536   etc
initial_batch_size = parameters.initial_batch_size

robust_bool = False

autoencoder_resnet_bool = parameters.autoencoder_resnet_bool
autoencoder_densenet_bool = False
autoencoder_resnet_concatenate_bool = False

autoencoder_unet_bool = parameters.autoencoder_unet_bool
autoencoder_unet_concatenate_bool = True

down_stride_bool = parameters.down_stride_bool
down_max_pool_too_bool = parameters.down_max_pool_too_bool
down_max_pool_too_concatenate_bool = True

cnn_only_autoencoder_bool = parameters.cnn_only_autoencoder_bool
dv_bool = parameters.dv_bool
pca_only_classifier_bool = False

classifier_resnet_bool = False
classifier_densenet_bool = False

up_stride_bool = parameters.up_stride_bool
up_upsample_too_bool = parameters.up_upsample_too_bool
up_upsample_too_concatenate_bool = True

boosted_model_paths = []


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(string):
    print("atoi")

    return int(string) if string.isdigit() else string


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_sorting(string):
    print("human_sorting")

    return [atoi(c) for c in re.split(r'(\d+)', string)]


def get_full_y_data():
    print("get_data")

    try:
        y_path = "{0}y/".format(data_path)

        y_file = "{0}{1}".format(y_path, os.listdir(y_path)[0])

        y = [[], []]

        with open(y_file, "r") as file:
            for line in file:
                line_split = line.strip().split("\t")

                y[0].append(line_split[0])
                y[1].append(line_split[1])

        y = np.asfarray(y)
    except:
        y = None

    return y


def get_train_data():
    print("get_train_data")

    window_size = 20
    step_size = 1
    test_split = 0.1

    x_path = "{0}x/".format(data_path)

    x_files = os.listdir(x_path)
    x_files.sort(key=human_sorting)
    x_files = ["{0}{1}".format(x_path, s) for s in x_files]

    y_path = "{0}y/".format(data_path)

    y_file = "{0}{1}".format(y_path, os.listdir(y_path)[0])

    y_values = [[], []]

    with open(y_file, "r") as file:
        for line in file:
            line_split = line.strip().split("\t")

            y_values[0].append(line_split[0])
            y_values[1].append(line_split[1])

    test_split_position = int(len(x_files) * ((test_split * -1.0) + 1.0))

    x_files_train = x_files[:test_split_position]
    x_files_test = x_files[test_split_position:]

    y_values_train = [y_values[0][:test_split_position], y_values[1][:test_split_position]]
    y_values_test = [y_values[0][test_split_position:], y_values[1][test_split_position:]]

    x_train_nii = []
    y_train = []

    x_files_train_len = len(x_files_train)

    for i in range(0, x_files_train_len - window_size, step_size):
        current_window_x = []
        current_window_y = []

        for j in range(i, i + window_size):
            current_window_x.append(x_files_train[j])
            current_window_y.append([y_values_train[0][j], y_values_train[1][j]])

        x_train_nii.append(current_window_x)
        y_train.append(current_window_y)

    current_window_x = []
    current_window_y = []

    for j in range(x_files_train_len - window_size, x_files_train_len):
        current_window_x.append(x_files_train[j])
        current_window_y.append([y_values_train[0][j], y_values_train[1][j]])

    x_train_nii.append(current_window_x)
    y_train.append(current_window_y)

    x_test_nii = []
    y_test = []

    for i in range(0, len(x_files_test) - window_size, step_size):
        current_window_x = []
        current_window_y = []

        for j in range(i, i + window_size):
            current_window_x.append(x_files_test[j])

            current_window_y.append([y_values_test[0][j], y_values_test[1][j]])

        x_test_nii.append(current_window_x)

        y_test.append(current_window_y)

    if step_size > 1:
        current_window_x = []
        current_window_y = []

        for j in range(len(x_files_test) - window_size, len(x_files_test)):
            current_window_x.append(x_files_test[j])

            current_window_y.append([y_values_test[0][j], y_values_test[1][j]])

        x_test_nii.append(current_window_x)

        y_test.append(current_window_y)

    x_train = []

    if not os.path.exists("{0}/x_train".format(output_path)):
        os.makedirs("{0}/x_train".format(output_path), mode=0o770)

    for i in range(len(x_train_nii)):
        window_volumes = []

        for j in range(len(x_train_nii[i])):
            current_volume = nibabel.load(x_train_nii[i][j]).get_data()

            window_volumes.append(current_volume)

        window_volumes = np.expand_dims(np.asfarray(window_volumes), -1)

        np.save("{0}/x_train/{1}.npy".format(output_path, str(i)), window_volumes)

        x_train.append("{0}/x_train/{1}.npy".format(output_path, str(i)))

    x_test = []

    if not os.path.exists("{0}/x_test".format(output_path)):
        os.makedirs("{0}/x_test".format(output_path), mode=0o770)

    for i in range(len(x_test_nii)):
        window_volumes = []

        for j in range(len(x_test_nii[i])):
            current_volume = nibabel.load(x_test_nii[i][j]).get_data()

            window_volumes.append(current_volume)

        window_volumes = np.expand_dims(np.asfarray(window_volumes), -1)

        np.save("{0}/x_test/{1}.npy".format(output_path, str(i)), window_volumes)

        x_test.append("{0}/x_test/{1}.npy".format(output_path, str(i)))

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    y_train = np.asfarray(y_train)
    y_test = np.asfarray(y_test)

    return x_train, y_train, x_test, y_test


def get_predict_data():
    print("get_predict_data")

    window_size = 20
    step_size = 1

    x_path = "{0}x/".format(data_path)

    x_files = os.listdir(x_path)
    x_files.sort(key=human_sorting)
    x_files = ["{0}{1}".format(x_path, s) for s in x_files]

    x = []
    window_index = []

    x_files_len = len(x_files)

    for i in range(0, x_files_len - window_size, step_size):
        current_window_x = []
        current_window_index = []

        for j in range(i, i + window_size):
            current_window_x.append(x_files[j])
            current_window_index.append(j)

        x.append(current_window_x)
        window_index.append(current_window_index)

    current_window_x = []
    current_window_index = []

    for j in range(x_files_len - window_size, x_files_len):
        current_window_x.append(x_files[j])
        current_window_index.append(j)

        x.append(current_window_x)
        window_index.append(current_window_index)

    x = np.asarray(x)

    x_predict_nii = x

    x_predict_nii = list(x_predict_nii.tolist())

    x_predict = []

    if not os.path.exists("{0}/x_predict".format(output_path)):
        os.makedirs("{0}/x_predict".format(output_path), mode=0o770)

    for i in range(len(x_predict_nii)):
        window_volumes = []

        for j in range(len(x_predict_nii[i])):
            current_volume = nibabel.load(x_predict_nii[i][j]).get_data()

            window_volumes.append(current_volume)

        window_volumes = np.expand_dims(np.asfarray(window_volumes), -1)

        np.save("{0}/x_predict/{1}.npy".format(output_path, str(i)), window_volumes)

        x_predict.append("{0}/x_predict/{1}.npy".format(output_path, str(i)))

    x_predict = np.asarray(x_predict)

    return x_predict, window_index


def copy_data(file_paths, new_output_path):
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path, mode=0o770)

    new_file_paths = []

    for i in range(len(file_paths)):
        data = np.load(file_paths[i], allow_pickle=True)
        np.save("{0}/{1}.npy".format(new_output_path, str(i)), data)

        new_file_paths.append("{0}/{1}.npy".format(new_output_path, str(i)))

    new_file_paths = np.asarray(new_file_paths)

    return new_file_paths


def train_save_preprocessing(x_train, y_train_output_1, x_test, y_test_output_1, y_train_output_2, y_test_output_2):
    if not os.path.exists("{0}_preprocessing".format(output_path)):
        os.makedirs("{0}_preprocessing".format(output_path), mode=0o770)

    if not os.path.exists("{0}_preprocessing/x_train".format(output_path)):
        os.makedirs("{0}_preprocessing/x_train".format(output_path), mode=0o770)

    for i in range(len(x_train)):
        data = np.load(x_train[i], allow_pickle=True)
        np.save("{0}/{1}.npy".format("{0}_preprocessing/x_train".format(output_path), str(i)), data)

    np.save("{0}/y_train_output_1.npy".format("{0}_preprocessing".format(output_path)), y_train_output_1)

    if not os.path.exists("{0}_preprocessing/x_test".format(output_path)):
        os.makedirs("{0}_preprocessing/x_test".format(output_path), mode=0o770)

    for i in range(len(x_test)):
        data = np.load(x_test[i], allow_pickle=True)
        np.save("{0}/{1}.npy".format("{0}_preprocessing/x_test".format(output_path), str(i)), data)

    np.save("{0}/y_test_output_1.npy".format("{0}_preprocessing".format(output_path)), y_test_output_1)

    if not os.path.exists("{0}_preprocessing/y_train_output_2".format(output_path)):
        os.makedirs("{0}_preprocessing/y_train_output_2".format(output_path), mode=0o770)

    for i in range(len(y_train_output_2)):
        data = np.load(y_train_output_2[i], allow_pickle=True)
        np.save("{0}/{1}.npy".format("{0}_preprocessing/y_train_output_2".format(output_path), str(i)), data)

    if not os.path.exists("{0}_preprocessing/y_test_output_2".format(output_path)):
        os.makedirs("{0}_preprocessing/y_test_output_2".format(output_path), mode=0o770)

    for i in range(len(y_test_output_2)):
        data = np.load(y_test_output_2[i], allow_pickle=True)
        np.save("{0}/{1}.npy".format("{0}_preprocessing/y_test_output_2".format(output_path), str(i)), data)

    with open("{0}/cropping".format(output_path), 'rb') as load_file:
        with open("{0}_preprocessing/cropping".format(output_path), 'wb') as save_file:
            pickle.dump(pickle.load(load_file), save_file)

    return True


def train_load_preprocessing():
    x_train = os.listdir("{0}_preprocessing/x_train/".format(output_path))
    x_train.sort(key=human_sorting)
    x_train = ["{0}{1}".format("{0}_preprocessing/x_train/".format(output_path), s) for s in x_train]

    y_train_output_1 = \
        np.load("{0}/y_train_output_1.npy".format("{0}_preprocessing".format(output_path)), allow_pickle=True)

    x_test = os.listdir("{0}_preprocessing/x_test/".format(output_path))
    x_test.sort(key=human_sorting)
    x_test = ["{0}{1}".format("{0}_preprocessing/x_test/".format(output_path), s) for s in x_test]

    y_test_output_1 = \
        np.load("{0}/y_test_output_1.npy".format("{0}_preprocessing".format(output_path)), allow_pickle=True)

    y_train_output_2 = os.listdir("{0}_preprocessing/y_train_output_2/".format(output_path))
    y_train_output_2.sort(key=human_sorting)
    y_train_output_2 = \
        ["{0}{1}".format("{0}_preprocessing/y_train_output_2/".format(output_path), s) for s in y_train_output_2]

    y_test_output_2 = os.listdir("{0}_preprocessing/y_test_output_2/".format(output_path))
    y_test_output_2.sort(key=human_sorting)
    y_test_output_2 = \
        ["{0}{1}".format("{0}_preprocessing/y_test_output_2/".format(output_path), s) for s in y_test_output_2]

    with open("{0}_preprocessing/cropping".format(output_path), 'rb') as load_file:
        with open("{0}/cropping".format(output_path), 'wb') as save_file:
            pickle.dump(pickle.load(load_file), save_file)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train_output_2 = np.asarray(y_train_output_2)
    y_test_output_2 = np.asarray(y_test_output_2)

    return x_train, y_train_output_1, x_test, y_test_output_1, y_train_output_2, y_test_output_2


def use_save_preprocessing(x_predict, window_index):
    if not os.path.exists("{0}_preprocessing".format(output_path)):
        os.makedirs("{0}_preprocessing".format(output_path), mode=0o770)

    if not os.path.exists("{0}_preprocessing/x_predict".format(output_path)):
        os.makedirs("{0}_preprocessing/x_predict".format(output_path), mode=0o770)

    for i in range(len(x_predict)):
        data = np.load(x_predict[i], allow_pickle=True)
        np.save("{0}/{1}.npy".format("{0}_preprocessing/x_predict".format(output_path), str(i)), data)

    if not os.path.exists("{0}_preprocessing/window_index".format(output_path)):
        os.makedirs("{0}_preprocessing/window_index".format(output_path), mode=0o770)

    np.save("{0}/window_index.npy".format("{0}_preprocessing/window_index".format(output_path)), window_index)

    with open("{0}/cropping".format(output_path), 'rb') as load_file:
        with open("{0}_preprocessing/cropping".format(output_path), 'wb') as save_file:
            pickle.dump(pickle.load(load_file), save_file)

    return True


def use_load_preprocessing():
    x_predict = os.listdir("{0}_preprocessing/x_predict/".format(output_path))
    x_predict.sort(key=human_sorting)
    x_predict = ["{0}{1}".format("{0}_preprocessing/x_predict/".format(output_path), s) for s in x_predict]

    window_index = \
        np.load("{0}/window_index.npy".format("{0}_preprocessing/window_index".format(output_path)), allow_pickle=True)

    with open("{0}_preprocessing/cropping".format(output_path), 'rb') as load_file:
        with open("{0}/cropping".format(output_path), 'wb') as save_file:
            pickle.dump(pickle.load(load_file), save_file)

    x_predict = np.asarray(x_predict)

    return x_predict, window_index


def get_preprocessed_train_data():
    # if preprocessed data already exist, use it
    if save_preprocessing_bool and os.path.exists("{0}_preprocessing".format(output_path)):
        x_train, y_train_output_1, x_test, y_test_output_1, y_train_output_2, y_test_output_2 = \
            train_load_preprocessing()
    else:
        x_train, y_train_output_1, x_test, y_test_output_1 = get_train_data()

        if os.path.exists("{0}/cropping".format(output_path)):
            with open("{0}/cropping".format(output_path), 'rb') as file:
                cropping = pickle.load(file)
        else:
            cropping = None

        x_train, cropping = preprocessing.data_crop(x_train, cropping)
        x_test, _ = preprocessing.data_crop(x_test, cropping)

        with open("{0}/cropping".format(output_path), 'wb') as file:
            pickle.dump(cropping, file)

        x_train = preprocessing.pad_data_upsample(x_train, cluster_bool)
        x_test = preprocessing.pad_data_upsample(x_test, cluster_bool)

        if not cluster_bool:
            x_train, _ = preprocessing.data_crop(x_train, ((0, 0), (32, 32), (0, 0)))
            x_test, _ = preprocessing.data_crop(x_test, ((0, 0), (32, 32), (0, 0)))

            x_train = preprocessing.data_downsample(x_train, cluster_bool, 4)
            x_test = preprocessing.data_downsample(x_test, cluster_bool, 4)

        y_train_output_2 = copy_data(x_train, "{0}/y_train_output_2".format(output_path))
        y_test_output_2 = copy_data(x_test, "{0}/y_test_output_2".format(output_path))

        x_train = preprocessing.data_preprocessing(x_train, robust_bool)
        x_test = preprocessing.data_preprocessing(x_test, robust_bool)

        if save_preprocessing_bool:
            train_save_preprocessing(x_train, y_train_output_1, x_test, y_test_output_1, y_train_output_2,
                                     y_test_output_2)

    return x_train, y_train_output_1, x_test, y_test_output_1, y_train_output_2, y_test_output_2


def get_evaluation_score(x_test, y_test_output_2, y_test_output_1, model):
    test_epoch_size = int(np.floor(x_test.shape[0] / 1))

    print("Test epoch size:\t{0}".format(str(test_epoch_size)))

    score = []

    for i in range(test_epoch_size):
        batch_indices = np.arange(i * 1, (i + 1) * 1, dtype=int)

        x_test_iteration = []

        for j in range(len(x_test[batch_indices])):
            x_test_iteration.append(np.load(x_test[batch_indices][j], allow_pickle=True))

        x_test_iteration = np.asfarray(x_test_iteration)

        y_test_output_2_iteration = []

        for j in range(len(y_test_output_2[batch_indices])):
            y_test_output_2_iteration.append(np.load(y_test_output_2[batch_indices][j], allow_pickle=True))

        y_test_output_2_iteration = np.asfarray(y_test_output_2_iteration)

        if float_sixteen_bool:
            x_test_iteration = x_test_iteration.astype(np.float16)
            y_test_output_1 = y_test_output_1.astype(np.float16)
            y_test_output_2_iteration = y_test_output_2_iteration.astype(np.float16)
        else:
            x_test_iteration = x_test_iteration.astype(np.float32)
            y_test_output_1 = y_test_output_1.astype(np.float32)
            y_test_output_2_iteration = y_test_output_2_iteration.astype(np.float32)

        score.append(model.test_on_batch(x_test_iteration,
                                         {"output_1": y_test_output_1[batch_indices],
                                          "output_2": y_test_output_2_iteration},
                                         reset_metrics=False))

    return score


def plot_classifier(plot_model_prediction_test, y_test_output_1, iteration):
    plot_model_prediction_test_output_1 = plot_model_prediction_test[0]
    plot_model_prediction_test_output_1 = np.squeeze(plot_model_prediction_test_output_1)
    plot_model_prediction_test_output_1 = plot_model_prediction_test_output_1.astype(np.float32)

    plot_y_test_output_1 = np.squeeze(y_test_output_1[0])
    plot_y_test_output_1 = plot_y_test_output_1.astype(np.float32)

    output_1_correlation, _ = scipy.stats.pearsonr(plot_y_test_output_1[:, 0],
                                                   plot_model_prediction_test_output_1[:, 0])

    if output_1_correlation < 0.0:
        plot_y_test_output_1[:, 0] = plot_y_test_output_1[:, 0] * -1.0

    output_1_correlation, _ = scipy.stats.pearsonr(plot_y_test_output_1[:, 1],
                                                   plot_model_prediction_test_output_1[:, 1])

    if output_1_correlation < 0.0:
        plot_y_test_output_1[:, 1] = plot_y_test_output_1[:, 1] * -1.0

    plot_y_test_output_1[:, 0] = plot_y_test_output_1[:, 0] - np.mean(plot_y_test_output_1[:, 0])
    plot_y_test_output_1[:, 1] = plot_y_test_output_1[:, 1] - np.mean(plot_y_test_output_1[:, 1])

    plot_model_prediction_test_output_1[:, 0] = \
        plot_model_prediction_test_output_1[:, 0] - np.mean(plot_model_prediction_test_output_1[:, 0])
    plot_model_prediction_test_output_1[:, 1] = \
        plot_model_prediction_test_output_1[:, 1] - np.mean(plot_model_prediction_test_output_1[:, 1])

    plot_y_test_output_1[:, 0] = plot_y_test_output_1[:, 0] / np.std(plot_y_test_output_1[:, 0])
    plot_y_test_output_1[:, 1] = plot_y_test_output_1[:, 1] / np.std(plot_y_test_output_1[:, 1])

    plot_model_prediction_test_output_1[:, 0] = \
        plot_model_prediction_test_output_1[:, 0] / np.std(plot_model_prediction_test_output_1[:, 0])
    plot_model_prediction_test_output_1[:, 1] = \
        plot_model_prediction_test_output_1[:, 1] / np.std(plot_model_prediction_test_output_1[:, 1])

    if plot_bool:
        plt.subplot(2, 1, 1)
        plt.plot(plot_y_test_output_1[:, 0])
        plt.plot(plot_model_prediction_test_output_1[:, 0])

        plt.subplot(2, 1, 2)
        plt.plot(plot_y_test_output_1[:, 1])
        plt.plot(plot_model_prediction_test_output_1[:, 1])

        plt.show()
        plt.close()

    if save_plot_bool:
        plt.subplot(2, 1, 1)
        plt.plot(plot_y_test_output_1[:, 0])
        plt.plot(plot_model_prediction_test_output_1[:, 0])

        plt.subplot(2, 1, 2)
        plt.plot(plot_y_test_output_1[:, 1])
        plt.plot(plot_model_prediction_test_output_1[:, 1])

        plt.savefig("{0}/{1}_plot_y_test_output_1".format(output_path, str(iteration)), format="png",
                    bbox_inches="tight")
        plt.close()

    return True


def train_model():
    print("train_model")

    # get data and lables
    x_train, y_train_output_1, x_test, y_test_output_1, y_train_output_2, y_test_output_2 = \
        get_preprocessed_train_data()

    print("Compute dtype:\t{0}".format(str(policy.compute_dtype)))
    print("Variable dtype:\t{0}".format(str(policy.variable_dtype)))
    print("{0}\ttrain samples".format(str(x_train.shape[0])))
    print("{0}\ttest samples".format(str(x_test.shape[0])))

    input_shape = np.load(x_train[0], allow_pickle=True).shape
    output_1_shape = y_train_output_1.shape[2]

    model = architecture.get_model(input_shape, output_1_shape)

    boosted_score = None

    if os.path.exists("{0}/boosted_score".format(output_path)):
        with open("{0}/boosted_score".format(output_path), 'rb') as file:
            boosted_score = pickle.load(file)

    if not os.path.exists("{0}/accuracy".format(output_path)):
        file = open("{0}/accuracy".format(output_path), "w")
        file.close()

        file = open("{0}/settings".format(output_path), "w")
        file.close()

        previous_max_iteration = 0

        iteration = 1
    else:
        if os.path.exists("{0}/settings".format(output_path)):
            with open("{0}/settings".format(output_path), "r") as file:
                for line in file:
                    line_split = line.strip().split("previous_max_iteration:\t")

                    if len(line_split) > 1:
                        previous_max_iteration = int(line_split[1])
        else:
            previous_max_iteration = 0

        max_accuracy = 0
        max_accuracy_iteration = 0
        max_iteration = 0

        with open("{0}/accuracy".format(output_path), "r") as file:
            for line in file:
                line_split = line.strip().split("\t")

                try:
                    current_iteration = int(line_split[0])
                    current_accuracy = float(line_split[1])

                    if current_iteration > previous_max_iteration:
                        if current_accuracy > max_accuracy:
                            max_accuracy = current_accuracy
                            max_accuracy_iteration = current_iteration

                        if current_iteration > max_iteration:
                            max_iteration = current_iteration
                except:
                    pass

        previous_max_iteration = max_iteration

        training_restart_accuracy_string = "Train model\nIteration selected:\t{0}\nPrevious max accuracy:\t{1}\nAutoencoder:\t{2}\nClassifier:\t{3}\n".format(str(max_accuracy), str(max_accuracy_iteration), str(autoencoder_bool), str(classifier_bool))

        print(training_restart_accuracy_string)

        with open("{0}/accuracy".format(output_path), "a") as file:
            file.write(training_restart_accuracy_string)

        model.load_weights("{0}/{1}".format(output_path, str(max_accuracy_iteration))).expect_partial()

        iteration = max_iteration + 1

    with open("{0}/settings".format(output_path), "w") as file:
        file.write("previous_max_iteration:\t{0}\n".format(str(previous_max_iteration)))

    model.summary()

    k.utils.plot_model(model, to_file="{0}/model.pdf".format(output_path), show_shapes=True, show_dtype=True,
                       show_layer_names=True, expand_nested=True)

    max_batch_size = initial_batch_size
    batch_size = initial_batch_size

    min_learning_rate = 1e-06
    learning_rate = initial_learning_rate

    callback_score = 0.0
    max_score_model = model.get_weights()

    callback_plateau_patience_iteration = 1
    callback_plateau_patience = 10
    callback_annealing_patience_iteration = 1
    callback_annealing_patience = 10

    if batch_size > max_batch_size:
        batch_size = max_batch_size

    train_byte_output_latch_bool = True
    train_gr = 0

    while True:
        print("Iteration:\t{0}".format(str(iteration)))

        if iteration > previous_max_iteration + 1:
            x_train_iteration = preprocessing.get_noisy_data(x_train, robust_bool, output_path)
            x_train_iteration, y_train_output_2_iteration = \
                preprocessing.get_transformed_data(x_train_iteration, y_train_output_2, robust_bool, output_path)

            train_epoch_size = int(np.floor(x_train.shape[0] / batch_size))

            print("Batch size:\t{0}".format(str(batch_size)))
            print("Train epoch size:\t{0}".format(str(train_epoch_size)))
            print("Learning rate:\t{0}".format(str(learning_rate)))

            for i in range(train_epoch_size):
                gr_batch_indices = []

                for j in range(batch_size):
                    gr_batch_indices.append(int(train_gr))

                    train_gr = train_gr + scipy.constants.golden

                    if train_gr > x_train.shape[0]:
                        train_gr = train_gr - x_train.shape[0]

                x_train_iteration_iteration = []

                for j in range(len(x_train_iteration[gr_batch_indices])):
                    x_train_iteration_iteration.append(np.load(x_train_iteration[gr_batch_indices][j],
                                                               allow_pickle=True))

                x_train_iteration_iteration = np.asfarray(x_train_iteration_iteration)

                y_train_output_2_iteration_iteration = []

                for j in range(len(y_train_output_2_iteration[gr_batch_indices])):
                    y_train_output_2_iteration_iteration.append(
                        np.load(y_train_output_2_iteration[gr_batch_indices][j], allow_pickle=True))

                y_train_output_2_iteration_iteration = np.asfarray(y_train_output_2_iteration_iteration)

                if float_sixteen_bool:
                    x_train_iteration_iteration = x_train_iteration_iteration.astype(np.float16)
                    y_train_output_1 = y_train_output_1.astype(np.float16)
                    y_train_output_2_iteration_iteration = y_train_output_2_iteration_iteration.astype(np.float16)
                else:
                    x_train_iteration_iteration = x_train_iteration_iteration.astype(np.float32)
                    y_train_output_1 = y_train_output_1.astype(np.float32)
                    y_train_output_2_iteration_iteration = y_train_output_2_iteration_iteration.astype(np.float32)

                if train_byte_output_latch_bool:
                    print("x_train_iteration_iteration size:\t{0}\ny_train_output_1 size:\t{1}\ny_train_output_2 size:\t{2}".format(hurry.filesize.size(x_train_iteration_iteration.nbytes), hurry.filesize.size(y_train_output_1.nbytes), hurry.filesize.size(y_train_output_2_iteration_iteration.nbytes)))

                    train_byte_output_latch_bool = False

                if boosted_score is not None:
                    loss = model.train_on_batch(x_train_iteration_iteration,
                                                {"output_1": y_train_output_1[gr_batch_indices],
                                                 "output_2": y_train_output_2_iteration_iteration},
                                                sample_weight=boosted_score[gr_batch_indices],
                                                reset_metrics=False)
                else:
                    loss = model.train_on_batch(x_train_iteration_iteration,
                                                {"output_1": y_train_output_1[gr_batch_indices],
                                                 "output_2": y_train_output_2_iteration_iteration},
                                                reset_metrics=False)

                print("{0:<5}/{1}:\tTrain output 1 loss:\t{2:<20}\tTrain output 2 loss:\t{3:<20}\tTrain output 1 accuracy:\t{4:<20}\tTrain output 2 accuracy:\t{5:<20}".format(str(i + 1), str(train_epoch_size), str(loss[1]), str(loss[2]), str(loss[3]), str(loss[4])))

        print()

        score = get_evaluation_score(x_test, y_test_output_2, y_test_output_1, model)
        score = np.mean(score, axis=0)

        output_1_accuracy = score[3]
        output_2_accuracy = score[4]

        print("Test output 1 loss:\t{0}".format(str(score[1])))
        print("Test output 2 loss:\t{0}".format(str(score[2])))
        print("Test output 1 accuracy:\t{0}".format(str(output_1_accuracy)))
        print("Test output 2 accuracy:\t{0}".format(str(output_2_accuracy)))

        print()

        if autoencoder_bool:
            if classifier_bool:
                new_callback_score = (output_1_accuracy + output_2_accuracy) / 2
            else:
                new_callback_score = output_2_accuracy

            if plot_bool or save_plot_bool:
                x_predict_iteration = preprocessing.get_noisy_data(np.expand_dims(x_test[0], 0), robust_bool,
                                                                   output_path)
                x_predict_iteration, _ = preprocessing.get_transformed_data(x_predict_iteration, None, robust_bool,
                                                                            output_path)
                x_predict_iteration = np.load(x_predict_iteration[0], allow_pickle=True)

                x_predict_iteration = np.asfarray(x_predict_iteration)

                plot_model_prediction_test = model.predict_on_batch(np.expand_dims(x_predict_iteration, 0))

                plot_model_prediction_test_output_2 = plot_model_prediction_test[1][0]
                plot_model_prediction_test_output_2 = np.squeeze(plot_model_prediction_test_output_2[0])
                plot_model_prediction_test_output_2 = plot_model_prediction_test_output_2.astype(np.float32)

                plot_x_predict_iteration = np.squeeze(x_predict_iteration[0])
                plot_x_predict_iteration = plot_x_predict_iteration.astype(np.float32)

                plot_y_test_output_2 = np.load(y_test_output_2[0], allow_pickle=True)
                plot_y_test_output_2 = np.squeeze(plot_y_test_output_2[0])
                plot_y_test_output_2 = plot_y_test_output_2.astype(np.float32)

                plot_y_test_output_2_min = np.min(plot_y_test_output_2)
                plot_y_test_output_2_std = np.std(plot_y_test_output_2)

                plot_model_prediction_test_output_2 = \
                    (((plot_model_prediction_test_output_2 - np.min(plot_model_prediction_test_output_2))
                      / np.std(plot_model_prediction_test_output_2)) * plot_y_test_output_2_std) \
                    + plot_y_test_output_2_min

                plot_x_predict_iteration = \
                    (((plot_x_predict_iteration - np.min(plot_x_predict_iteration))
                      / np.std(plot_x_predict_iteration)) * plot_y_test_output_2_std) \
                    + plot_y_test_output_2_min

                if plot_bool:
                    plt.imshow(plot_y_test_output_2[:, :, int(plot_y_test_output_2.shape[2] / 2)], cmap="Greys")
                    plt.show()
                    plt.close()

                    plt.imshow(plot_x_predict_iteration[:, :, int(plot_x_predict_iteration.shape[2] / 2)], cmap="Greys")
                    plt.show()
                    plt.close()

                    plt.imshow(plot_model_prediction_test_output_2[:, :,
                               int(plot_model_prediction_test_output_2.shape[2] / 2)], cmap="Greys")
                    plt.show()
                    plt.close()

                if save_plot_bool:
                    plt.imshow(plot_y_test_output_2[:, :, int(plot_y_test_output_2.shape[2] / 2)], cmap="Greys")
                    plt.savefig("{0}/{1}_plot_y_test_output_2".format(output_path, str(iteration)), format="png",
                                bbox_inches="tight")
                    plt.close()

                    plt.imshow(plot_x_predict_iteration[:, :, int(plot_x_predict_iteration.shape[2] / 2)], cmap="Greys")
                    plt.savefig("{0}/{1}_plot_x_predict_iteration".format(output_path, str(iteration)),
                                format="png", bbox_inches="tight")
                    plt.close()

                    plt.imshow(plot_model_prediction_test_output_2[:, :,
                               int(plot_model_prediction_test_output_2.shape[2] / 2)], cmap="Greys")
                    plt.savefig("{0}/{1}_plot_model_prediction_test_output_2".format(output_path, str(iteration)),
                                format="png", bbox_inches="tight")
                    plt.close()

                if classifier_bool:
                    plot_classifier(plot_model_prediction_test, y_test_output_1, iteration)
        else:
            if classifier_bool:
                new_callback_score = output_1_accuracy

                x_predict_iteration = np.load(x_test[0], allow_pickle=True)
                x_predict_iteration = np.asfarray(x_predict_iteration)

                plot_model_prediction_test = model.predict_on_batch(np.expand_dims(x_predict_iteration, 0))

                plot_classifier(plot_model_prediction_test, y_test_output_1, iteration)
            else:
                new_callback_score = 0.0

        if iteration <= previous_max_iteration + 1:
            new_callback_score = 1e-05

        model.save_weights("{0}/{1}".format(output_path, str(iteration)))

        with open("{0}/accuracy".format(output_path), "a") as file:
            file.write("{0}\t{1}\n".format(str(iteration), str(new_callback_score)))

        if np.abs(new_callback_score - callback_score) < 1e-04:
            if callback_plateau_patience_iteration < callback_plateau_patience:
                callback_plateau_patience_iteration = callback_plateau_patience_iteration + 1
            else:
                if batch_size != max_batch_size:
                    batch_size = batch_size * 2

                    if batch_size > max_batch_size:
                        batch_size = max_batch_size
                else:
                    if learning_rate != learning_rate:
                        learning_rate = learning_rate * 0.1

                        if learning_rate < min_learning_rate:
                            learning_rate = min_learning_rate

                        k.backend.set_value(model.optimizer.learning_rate, learning_rate)

                callback_plateau_patience_iteration = 1
                callback_annealing_patience_iteration = 1
        else:
            callback_plateau_patience_iteration = 1

        if new_callback_score > callback_score:
            callback_score = new_callback_score
            max_score_model = model.get_weights()

            callback_annealing_patience_iteration = 1
        else:
            model.set_weights(max_score_model)

            if callback_annealing_patience_iteration < callback_annealing_patience:
                callback_annealing_patience_iteration = callback_annealing_patience_iteration + 1
            else:
                if batch_size != max_batch_size:
                    batch_size = batch_size * 2

                    if batch_size > max_batch_size:
                        batch_size = max_batch_size
                else:
                    if learning_rate != learning_rate:
                        learning_rate = learning_rate * 0.1

                        if learning_rate < min_learning_rate:
                            learning_rate = min_learning_rate

                        k.backend.set_value(model.optimizer.learning_rate, learning_rate)

                callback_plateau_patience_iteration = 1
                callback_annealing_patience_iteration = 1

        iteration = iteration + 1


def get_model_predictions(x_predict, model):
    batch_size = 1

    predict_epoch_size = int(np.floor(x_predict.shape[0] / batch_size))

    print("Predict Epoch size:\t{0}".format(str(predict_epoch_size)))

    current_model_predictions = []

    for j in range(predict_epoch_size):
        batch_indices = np.arange(j * batch_size, (j + 1) * batch_size, dtype=int)

        x_predict_iteration = np.load(x_predict[batch_indices[0]], allow_pickle=True)
        x_predict_iteration = np.asfarray(x_predict_iteration)

        if float_sixteen_bool:
            x_predict_iteration = x_predict_iteration.astype(np.float16)
        else:
            x_predict_iteration = x_predict_iteration.astype(np.float32)

        current_model_predictions.append(model.predict_on_batch(np.expand_dims(x_predict_iteration, 0)))

    return current_model_predictions


def use_model():
    print("use_model")

    if save_preprocessing_bool and os.path.exists("{0}_preprocessing/x_predict/".format(output_path)):
        x_predict, window_index = use_load_preprocessing()
    else:
        x_predict, window_index = get_predict_data()

        with open("{0}/cropping".format(output_path), 'rb') as file:
            cropping = pickle.load(file)

        x_predict, _ = preprocessing.data_crop(x_predict, cropping)
        x_predict = preprocessing.pad_data_upsample(x_predict, cluster_bool)

        if not cluster_bool:
            x_predict, _ = preprocessing.data_crop(x_predict, ((0, 0), (32, 32), (0, 0)))
            x_predict = preprocessing.data_downsample(x_predict, cluster_bool, 4)

        x_predict = preprocessing.data_preprocessing(x_predict, robust_bool)

        use_save_preprocessing(x_predict, window_index)

    print("{0}\tpredict samples".format(str(x_predict.shape[0])))

    input_shape = np.load(x_predict[0], allow_pickle=True).shape
    output_1_shape = 2

    model = architecture.get_model(input_shape, output_1_shape)

    model.summary()

    boosted_model_paths_len = len(boosted_model_paths)

    if boosted_model_paths_len > 0:
        x_train, y_train_output_1, x_test, y_test_output_1, y_train_output_2, y_test_output_2 = \
            get_preprocessed_train_data()

        boosted_score = None
        total_score = 0
        model_predictions = None

        for i in range(boosted_model_paths_len):
            model.load_weights(boosted_model_paths[i]).expect_partial()

            score = get_evaluation_score(x_train, y_train_output_2, y_train_output_1, model)

            if autoencoder_bool:
                if classifier_bool:
                    score = np.mean(np.asfarray(score)[:, 3:5], axis=1)
                else:
                    score = np.asfarray(score)[:, 4]
            else:
                if classifier_bool:
                    score = np.asfarray(score)[:, 3:5]
                else:
                    score = 0.0

            if boosted_score is not None:
                boosted_score = boosted_score + np.asfarray(score)
            else:
                boosted_score = np.asfarray(score)

            score = get_evaluation_score(x_test, y_test_output_2, y_test_output_1, model)

            if autoencoder_bool:
                if classifier_bool:
                    score = np.mean(np.asfarray(score)[:, 3:5])
                else:
                    score = np.mean(np.asfarray(score)[:, 4])
            else:
                if classifier_bool:
                    score = np.mean(np.asfarray(score)[:, 3])
                else:
                    score = 0.0

            if model_predictions is not None:
                current_model_predictions = get_model_predictions(x_predict, model)

                for j in range(len(model_predictions)):
                    model_predictions[j][0] = model_predictions[j][0] + (current_model_predictions[j][0] * score)
                    model_predictions[j][1] = model_predictions[j][1] + (current_model_predictions[j][1] * score)
            else:
                model_predictions = get_model_predictions(x_predict, model)

                for j in range(len(model_predictions)):
                    model_predictions[j][0] = model_predictions[j][0] * score
                    model_predictions[j][1] = model_predictions[j][1] * score

            total_score = total_score + score

        boosted_score = boosted_score / boosted_model_paths_len
        boosted_score = boosted_score * -1.0
        boosted_score = (boosted_score - np.min(boosted_score)) + 1.0

        with open("{0}/boosted_score".format(output_path), 'wb') as file:
            pickle.dump(boosted_score, file)

        for j in range(len(model_predictions)):
            model_predictions[j][0] = model_predictions[j][0] / total_score
            model_predictions[j][1] = model_predictions[j][1] / total_score
    else:
        if os.path.exists("{0}/settings".format(output_path)):
            with open("{0}/settings".format(output_path), "r") as file:
                for line in file:
                    line_split = line.strip().split("previous_max_iteration:\t")

                    if len(line_split) > 1:
                        previous_max_iteration = int(line_split[1])
        else:
            previous_max_iteration = 0

        max_accuracy = 0
        max_accuracy_iteration = 0
        max_iteration = 0

        with open("{0}/accuracy".format(output_path), "r") as file:
            for line in file:
                line_split = line.strip().split("\t")

                try:
                    current_iteration = int(line_split[0])
                    current_accuracy = float(line_split[1])

                    if current_iteration > previous_max_iteration:
                        if current_accuracy > max_accuracy:
                            max_accuracy = current_accuracy
                            max_accuracy_iteration = current_iteration

                        if current_iteration > max_iteration:
                            max_iteration = current_iteration
                except:
                    pass

        training_restat_accuracy_string = "Use model\nIteration selected:\t{0}\nPrevious max accuracy:\t{1}\nAutoencoder:\t{2}\nClassifier:\t{3}".format(str(max_accuracy), str(max_accuracy_iteration), str(autoencoder_bool), str(classifier_bool))

        print(training_restat_accuracy_string)

        model.load_weights("{0}/{1}".format(output_path, str(max_accuracy_iteration))).expect_partial()

        model_predictions = get_model_predictions(x_predict, model)

    for i in range(1, len(model_predictions)):
        if float_sixteen_bool:
            output_1_1_correlation, _ = \
                scipy.stats.pearsonr(np.ravel(np.asfarray(model_predictions[i - 1][0][0, :, 0], dtype=np.float32)),
                                     np.ravel(np.asfarray(model_predictions[i][0][0, :, 0], dtype=np.float32)))
        else:
            output_1_1_correlation, _ = scipy.stats.pearsonr(np.ravel(model_predictions[i - 1][0][0, :, 0]),
                                                             np.ravel(model_predictions[i][0][0, :, 0]))

        if output_1_1_correlation < 0.0:
            model_predictions[i][0][0, :, 0] = model_predictions[i][0][0, :, 0] * -1.0

        if float_sixteen_bool:
            output_1_2_correlation, _ = \
                scipy.stats.pearsonr(np.ravel(np.asfarray(model_predictions[i - 1][0][0, :, 1], dtype=np.float32)),
                                     np.ravel(np.asfarray(model_predictions[i][0][0, :, 1], dtype=np.float32)))
        else:
            output_1_2_correlation, _ = scipy.stats.pearsonr(np.ravel(model_predictions[i - 1][0][0, :, 1]),
                                                             np.ravel(model_predictions[i][0][0, :, 1]))

        if output_1_2_correlation < 0.0:
            model_predictions[i][0][0, :, 1] = model_predictions[i][0][0, :, 1] * -1.0

    model_predictions_output_1 = []
    model_predictions_output_2 = []

    for i in range(np.max(window_index) + 1):
        model_predictions_output_1.append(np.asfarray(model_predictions[0][0][0, 0], dtype=np.float64).copy())
        model_predictions_output_2.append(np.asfarray(np.squeeze(model_predictions[0][1][0, 0]), dtype=np.float64).copy())

    for i in range(len(window_index)):
        for j in range(len(window_index[i])):
            model_predictions_output_1[window_index[i][j]] = \
                model_predictions_output_1[window_index[i][j]] + model_predictions[i][0][0, j]

            model_predictions_output_2[window_index[i][j]] = \
                model_predictions_output_2[window_index[i][j]] + np.squeeze(model_predictions[i][1][0, j])

    for i in range(len(model_predictions_output_1)):
        model_predictions_output_1[i] = \
            model_predictions_output_1[i] / sum(np.count_nonzero(nested_list == i) for nested_list in window_index)

        model_predictions_output_2[i] = \
            model_predictions_output_2[i] / sum(np.count_nonzero(nested_list == i) for nested_list in window_index)

    with open("{0}/model_prediction".format(output_path), 'wb') as file:
        pickle.dump([model_predictions_output_1, model_predictions_output_2], file)

    if plot_bool:
        y = get_full_y_data()

        if autoencoder_bool:
            plot_model_prediction = model_predictions_output_2[0]
            plot_model_prediction = plot_model_prediction.astype(np.float32)

            plt.imshow(plot_model_prediction[:, :, int(plot_model_prediction.shape[2] / 2)], cmap="Greys")
            plt.show()
            plt.close()

        if classifier_bool:
            plot_model_prediction = np.asfarray(model_predictions_output_1)
            plot_model_prediction = plot_model_prediction.astype(np.float32)

            if y is not None:
                plot_y_iteration = np.transpose(y)
                plot_y_iteration = plot_y_iteration.astype(np.float32)

                plot_y_iteration_0_correlation, _ = scipy.stats.pearsonr(plot_y_iteration[:, 0],
                                                                         plot_model_prediction[:, 0])

                if plot_y_iteration_0_correlation < 0.0:
                    plot_model_prediction[:, 0] = plot_model_prediction[:, 0] * -1.0

                plot_y_iteration_1_correlation, _ = scipy.stats.pearsonr(plot_y_iteration[:, 1],
                                                                         plot_model_prediction[:, 1])

                if plot_y_iteration_1_correlation < 0.0:
                    plot_model_prediction[:, 1] = plot_model_prediction[:, 1] * -1.0

                plot_y_iteration[:, 0] = plot_y_iteration[:, 0] - np.mean(plot_y_iteration[:, 0])
                plot_y_iteration[:, 1] = plot_y_iteration[:, 1] - np.mean(plot_y_iteration[:, 1])

                plot_model_prediction[:, 0] = plot_model_prediction[:, 0] - np.mean(plot_model_prediction[:, 0])
                plot_model_prediction[:, 1] = plot_model_prediction[:, 1] - np.mean(plot_model_prediction[:, 1])

                plot_y_iteration[:, 0] = plot_y_iteration[:, 0] / np.std(plot_y_iteration[:, 0])
                plot_y_iteration[:, 1] = plot_y_iteration[:, 1] / np.std(plot_y_iteration[:, 1])

                plot_model_prediction[:, 0] = plot_model_prediction[:, 0] / np.std(plot_model_prediction[:, 0])
                plot_model_prediction[:, 1] = plot_model_prediction[:, 1] / np.std(plot_model_prediction[:, 1])

                plt.subplot(2, 1, 1)
                plt.plot(plot_y_iteration[:, 0])
                plt.plot(plot_model_prediction[:, 0])

                plt.subplot(2, 1, 2)
                plt.plot(plot_y_iteration[:, 1])
                plt.plot(plot_model_prediction[:, 1])

                plt.show()
                plt.close()

                if save_plot_bool:
                    plt.subplot(2, 1, 1)
                    plt.plot(plot_y_iteration[:, 0])
                    plt.plot(plot_model_prediction[:, 0])

                    plt.subplot(2, 1, 2)
                    plt.plot(plot_y_iteration[:, 1])
                    plt.plot(plot_model_prediction[:, 1])

                    plt.savefig("{0}/full_plot_model_prediction".format(output_path), format="png",
                                bbox_inches="tight")
                    plt.close()
            else:
                plt.subplot(2, 1, 1)
                plt.plot(plot_model_prediction[:, 0])

                plt.subplot(2, 1, 2)
                plt.plot(plot_model_prediction[:, 1])

                plt.show()
                plt.close()

                if save_plot_bool:
                    plt.subplot(2, 1, 1)
                    plt.plot(plot_model_prediction[:, 0])

                    plt.subplot(2, 1, 2)
                    plt.plot(plot_model_prediction[:, 1])

                    plt.savefig("{0}/full_plot_model_prediction".format(output_path), format="png",
                                bbox_inches="tight")
                    plt.close()

    return True


def main():
    print("main")

    # if debugging, remove previous output directory
    if debug_bool:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

    # creare output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o770)

    # create log file and begin writing to it
    logfile_path = "{0}/logfile.log".format(output_path)

    if os.path.exists(logfile_path):
        os.remove(logfile_path)

    transcript.start(logfile_path)

    if train_bool:
        train_model()
    else:
        use_model()

    transcript.stop()

    return True


if __name__ == "__main__":
    main()
