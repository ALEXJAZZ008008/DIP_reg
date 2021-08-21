# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import re
import shutil
import random
import numpy as np
import scipy.constants
import scipy.stats
import scipy.ndimage
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
import nibabel

import parameters
import transcript
import preprocessing
import architecture


random.seed()


float_sixteen_bool = True  # set the network to use float16 data
cpu_bool = False  # if using CPU, set to true: disables mixed precision computation

# mixed precision float16 computation allows the network to use both float16 and float32 where necessary,
# this improves performance on the GPU.
if float_sixteen_bool and not cpu_bool:
    policy = k.mixed_precision.Policy("mixed_float16")
    k.mixed_precision.set_global_policy(policy)
else:
    policy = k.mixed_precision.Policy(tf.dtypes.float32.name)
    k.mixed_precision.set_global_policy(policy)

plot_bool = False
save_plot_bool = True
data_path = "{0}/DIP_RDP_data/xcat/".format(os.path.dirname(os.getcwd()))
output_path = "{0}/output".format(os.getcwd())
data_window_size = 47
data_window_bool = False

robust_bool = False

autoencoder_resnet_bool = parameters.autoencoder_resnet_bool
autoencoder_densenet_bool = False
autoencoder_resnet_concatenate_bool = False

autoencoder_unet_bool = parameters.autoencoder_unet_bool
autoencoder_unet_concatenate_bool = parameters.autoencoder_unet_concatenate_bool

down_stride_bool = parameters.down_stride_bool
down_pool_too_bool = parameters.down_pool_too_bool
down_max_pool_too_concatenate_bool = parameters.down_max_pool_too_concatenate_bool

up_stride_bool = parameters.up_stride_bool
up_upsample_too_bool = parameters.up_upsample_too_bool
up_upsample_too_concatenate_bool = parameters.up_upsample_too_concatenate_bool


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(string):
    print("atoi")

    return int(string) if string.isdigit() else string


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_sorting(string):
    print("human_sorting")

    return [atoi(c) for c in re.split(r'(\d+)', string)]


def get_data_windows(data):
    print("get_data_windows")

    if data_window_bool:
        pass
    else:
        axial_size = data.shape[-1]

        if axial_size > data_window_size:
            data_centre = int(axial_size / 2.0)

            data_upper_window_size = int(data_window_size / 2.0)
            data_lower_window_size = data_upper_window_size

            if data_upper_window_size + data_lower_window_size < data_window_size:
                data_upper_window_size = data_upper_window_size + 1

            data = data[:, :, data_centre - data_upper_window_size:data_centre + data_lower_window_size]

    return data


def get_train_data():
    print("get_train_data")

    y_path = "{0}y/".format(data_path)

    y_files = os.listdir(y_path)
    y_files.sort(key=human_sorting)
    y_files = ["{0}{1}".format(y_path, s) for s in y_files]

    x = []
    y = []

    if not os.path.exists("{0}/x_train".format(output_path)):
        os.makedirs("{0}/x_train".format(output_path), mode=0o770)

    if not os.path.exists("{0}/y_train".format(output_path)):
        os.makedirs("{0}/y_train".format(output_path), mode=0o770)

    for i in range(len(y_files)):
        current_volume = nibabel.load(y_files[i]).get_data()

        current_volume = get_data_windows(current_volume)

        np.save("{0}/y_train/{1}.npy".format(output_path, str(i)), current_volume)
        y.append("{0}/y_train/{1}.npy".format(output_path, str(i)))

        if parameters.noise_input:
            current_volume = np.random.normal(size=current_volume.shape)
        else:
            if parameters.smoothed_input:
                current_volume = scipy.ndimage.gaussian_filter(current_volume, sigma=1.5, mode="mirror")

        np.save("{0}/x_train/{1}.npy".format(output_path, str(i)), current_volume)
        x.append("{0}/x_train/{1}.npy".format(output_path, str(i)))

    y = np.asarray(y)

    gt_path = "{0}gt/".format(data_path)

    if os.path.exists(gt_path):
        gt_files = os.listdir(gt_path)
        gt_files.sort(key=human_sorting)
        gt_files = ["{0}{1}".format(gt_path, s) for s in gt_files]

        gt = []

        if not os.path.exists("{0}/gt_train".format(output_path)):
            os.makedirs("{0}/gt_train".format(output_path), mode=0o770)

        for i in range(len(gt_files)):
            current_volume = nibabel.load(gt_files[i]).get_data()

            current_volume = get_data_windows(current_volume)

            np.save("{0}/gt_train/{1}.npy".format(output_path, str(i)), current_volume)
            gt.append("{0}/gt_train/{1}.npy".format(output_path, str(i)))

        gt = np.asarray(gt)
    else:
        gt = None

    return x, y, gt


def get_preprocessed_train_data():
    print("get_preprocessed_train_data")

    x, y, gt = get_train_data()

    x = preprocessing.data_upsample(x)
    y = preprocessing.data_upsample(y)
    gt = preprocessing.data_upsample(gt)

    x = preprocessing.data_preprocessing(x, robust_bool)
    y = preprocessing.data_preprocessing(y, robust_bool)
    gt = preprocessing.data_preprocessing(gt, robust_bool)

    return x, y, gt


# https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(batch_size, model):
    print("get_model_memory_usage")

    shapes_mem_count = 0
    internal_model_mem_count = 0

    for l in model.layers:
        layer_type = l.__class__.__name__

        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)

        single_layer_mem = 1
        out_shape = l.output_shape

        if type(out_shape) is list:
            out_shape = out_shape[0]

        for s in out_shape:
            if s is None:
                continue

            single_layer_mem = single_layer_mem * s

        shapes_mem_count = shapes_mem_count + single_layer_mem

    trainable_count = np.sum([k.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([k.backend.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0

    if k.backend.floatx() == 'float16':
        number_size = 2.0

    if k.backend.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count

    return gbytes


def get_prediction(x, model):
    print("get_prediction")

    x_iteration = np.asarray([np.load(x[0], allow_pickle=True)])

    if float_sixteen_bool:
        x_iteration = x_iteration.astype(np.float16)
    else:
        x_iteration = x_iteration.astype(np.float32)

    x_prediction = model.predict_on_batch(x_iteration)

    return x_prediction


def get_evaluation_score(x_prediction, gt):
    print("get_evaluation_score")

    gt_iteration = np.asarray([np.load(gt[0], allow_pickle=True)])

    if float_sixteen_bool:
        x_prediction = x_prediction.astype(np.float32)

    accuracy = np.corrcoef(np.ravel(np.squeeze(gt_iteration)), np.ravel(np.squeeze(x_prediction)))[0, 1]

    return accuracy


def train_model():
    print("train_model")

    # get data and lables
    x, y, gt = get_preprocessed_train_data()

    input_shape = np.load(x[0], allow_pickle=True).shape

    print("Compute dtype:\t{0}".format(str(policy.compute_dtype)))
    print("Variable dtype:\t{0}".format(str(policy.variable_dtype)))
    print("{0}\ttrain shape".format(str(input_shape)))

    model = architecture.get_model(input_shape)
    model.summary()

    k.utils.plot_model(model, to_file="{0}/model.pdf".format(output_path), show_shapes=True, show_dtype=True,
                       show_layer_names=True, expand_nested=True)

    print("Memory usage:\t{0}".format(str(get_model_memory_usage(1, model))))

    iteration = 0

    while True:
        print("Iteration:\t{0}".format(str(iteration)))

        x_iteration = preprocessing.get_noisy_data(x, robust_bool, output_path)
        x_iteration, y_iteration = preprocessing.get_transformed_data(x_iteration, y, robust_bool, output_path)

        x_train_iteration = np.asarray([np.load(x_iteration[0], allow_pickle=True)])
        y_train_iteration = np.asarray([np.load(y_iteration[0], allow_pickle=True)])

        if float_sixteen_bool:
            x_train_iteration = x_train_iteration.astype(np.float16)
            y_train_iteration = y_train_iteration.astype(np.float16)
        else:
            x_train_iteration = x_train_iteration.astype(np.float32)
            y_train_iteration = y_train_iteration.astype(np.float32)

        loss = model.train_on_batch(x_train_iteration,
                                    {"output": y_train_iteration},
                                    reset_metrics=False)

        print("Loss:\t{0:<20}\tAccuracy:\t{1:<20}".format(str(loss[0]), str(loss[1])))

        if gt is not None or (plot_bool or save_plot_bool):
            x_prediction = get_prediction(x, model)
        else:
            x_prediction = None

        if gt is not None:
            gt_accuracy = get_evaluation_score(x_prediction, gt)

            print("GT accuracy:\t{0}".format(str(gt_accuracy)))

        if plot_bool or save_plot_bool:
            x_prediction = np.squeeze(x_prediction).astype(np.float64)

            y_plot = np.squeeze(np.asarray(np.load(y[0], allow_pickle=True)))

            if gt is not None:
                gt_plot = np.squeeze(np.asarray(np.load(gt[0], allow_pickle=True)))
            else:
                gt_plot = None

            if plot_bool:
                plt.figure()

                if gt is not None:
                    plt.subplot(1, 3, 1)
                else:
                    plt.subplot(1, 2, 1)

                plt.imshow(x_prediction[:, :, int(x_prediction.shape[2] / 2)], cmap="Greys")

                if gt is not None:
                    plt.subplot(1, 3, 2)
                else:
                    plt.subplot(1, 2, 2)

                plt.imshow(y_plot[:, :, int(y_plot.shape[2] / 2)], cmap="Greys")

                if gt_plot is not None:
                    plt.subplot(1, 3, 3)
                    plt.imshow(gt_plot[:, :, int(gt_plot.shape[2] / 2)], cmap="Greys")

                plt.show()
                plt.close()

            if save_plot_bool:
                plt.figure()

                if gt is not None:
                    plt.subplot(1, 3, 1)
                else:
                    plt.subplot(1, 2, 1)

                plt.imshow(x_prediction[:, :, int(x_prediction.shape[2] / 2)], cmap="Greys")

                if gt is not None:
                    plt.subplot(1, 3, 2)
                else:
                    plt.subplot(1, 2, 2)

                plt.imshow(y_plot[:, :, int(y_plot.shape[2] / 2)], cmap="Greys")

                if gt_plot is not None:
                    plt.subplot(1, 3, 3)
                    plt.imshow(gt_plot[:, :, int(gt_plot.shape[2] / 2)], cmap="Greys")

                plt.savefig("{0}/{1}".format(output_path, str(iteration)), format="png", bbox_inches="tight")
                plt.close()

        iteration = iteration + 1


def main():
    print("main")

    # if debugging, remove previous output directory
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

    train_model()

    transcript.stop()

    return True


if __name__ == "__main__":
    main()
