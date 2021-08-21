# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os

import numpy as np
import matplotlib.pylab as plt


input_path = "{0}/DIP_TV/output/logfile.log".format(os.path.dirname(os.getcwd()))

patient_time_point_split = "Patient/Time point:\t"
window_split = "Window:\t"

split = [["GT accuracy:\t"], ["Loss:\t", "\tAccuracy:\t"]]
split_position = [[-1], [-1, 0]]

max_bool = [True, False]

output_path = "{0}/output/".format(os.getcwd())
output_prefix = ["gt_accuracy", "loss"]

if not os.path.exists(output_path):
    os.makedirs(output_path, mode=0o770)

patient_time_points = []

with open(input_path, 'r') as file:
    for line in file:
        line = line.strip()

        split_array = line.split(patient_time_point_split)

        if len(split_array) >= 2:
            patient_time_points.append(int(split_array[-1]))

for i in range(len(patient_time_points)):
    windows = []

    with open(input_path, 'r') as file:
        patient_split_bool = False

        for line in file:
            line = line.strip()

            split_array = line.split(patient_time_point_split)

            if len(split_array) >= 2:
                if int(split_array[-1]) == patient_time_points[i]:
                    patient_split_bool = True
                else:
                    patient_split_bool = False

            if patient_split_bool:
                split_array = line.split(window_split)

                if len(split_array) >= 2:
                    windows.append(int(split_array[-1]))

    for j in range(len(windows)):
        for k in range(len(split)):
            output = []

            with open(input_path, 'r') as file:
                patient_split_bool = False
                window_split_bool = False

                for line in file:
                    line = line.strip()

                    split_array = line.split(patient_time_point_split)

                    if len(split_array) >= 2:
                        if int(split_array[-1]) == patient_time_points[i]:
                            patient_split_bool = True
                        else:
                            patient_split_bool = False

                    if patient_split_bool:
                        split_array = line.split(window_split)

                        if len(split_array) >= 2:
                            if int(split_array[-1]) == windows[j]:
                                window_split_bool = True
                            else:
                                window_split_bool = False

                        if window_split_bool:
                            split_bool = True

                            for l in range(len(split[k])):
                                split_array = line.split(split[k][l])

                                if len(split_array) >= 2:
                                    line = split_array[split_position[k][l]]
                                else:
                                    split_bool = False

                                    break

                            if split_bool:
                                output.append(float(line))

            output = np.asarray(output)

            print("Patient/Time point:\t{0}".format(str(i)))
            print(output_prefix[k])

            if max_bool[k]:
                print("Max value:\t{0:<20}\tMax index:\t{1:<20}".format(str(np.max(output)), str(np.argmax(output))))
            else:
                print("Min value:\t{0:<20}\tMax index:\t{1:<20}".format(str(np.min(output)), str(np.argmin(output))))

            output_gradient = np.gradient(output)

            patience = 10
            current_patience = 0

            output_gradient_len = len(output_gradient)
            plateau_index = output_gradient_len - 1

            for l in range(output_gradient_len):
                if np.isclose(output_gradient[l], 0.0, atol=1e-04):
                    current_patience = current_patience + 1

                    if current_patience >= patience:
                        plateau_index = l

                        break
                else:
                    current_patience = 0

            print("Plateau value:\t{0:<20}\tPlateau index:\t{1:<20}".format(str(output[plateau_index]), str(plateau_index)))

            plt.figure()
            plt.plot(output)
            plt.savefig("{0}/{1}_{2}_{3}.png".format(output_path, output_prefix[k], str(i), str(j)), format="png",
                        dpi=600, bbox_inches="tight")
            plt.close()
