# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os


data_path = "{0}/DIP_RDP_data/insert_inpainting_test/".format(os.path.dirname(os.getcwd()))
output_path = "{0}/output/insert_inpainting_test/".format(os.getcwd())

# gaussian_path = "{0}/input/untrained/".format(os.getcwd())
gaussian_path = ""

data_window_size = 32
data_window_bool = True

data_resample_power_of = 2

data_input_bool = True

data_gaussian_smooth_sigma_xy = 0.0
data_gaussian_smooth_sigma_z = 0.0

input_gaussian_weight = 1.0


# model_path = "{0}/input/untrained/".format(os.getcwd())
model_path = ""

# layer_layers = [2, 2, 2, 2, 2, 1]
# layer_depth = [1, 2, 4, 8, 16, 32]
# layer_groups = [1, 1, 1, 1, 1, 1]

layer_layers = [2, 2, 2, 2, 1]
layer_depth = [1, 2, 4, 8, 16]
layer_groups = [1, 1, 1, 1, 1]


new_model_patient_bool = False
new_optimiser_patient_bool = True

new_model_window_bool = False
new_optimiser_window_bool = True


jitter_magnitude = 6

elastic_jitter_bool = False
elastic_jitter_sigma = 0.0
elastic_jitter_points_iterations = 6


input_gaussian_sigma = 0.0
skip_gaussian_sigma = 0.0
layer_gaussian_sigma = 0.0


dropout = 0.1

bayesian_test_bool = False
bayesian_output_bool = True
bayesian_iterations = 64


log_cosh_weight = 1e00

relative_difference_bool = True
relative_difference_weight = 1e01
relative_difference_edge_preservation_weight = 1e00

l1_weight_activity = 1e01
l1_weight_prelu = 0.0
l2_weight = 1e-01

uncertainty_weight = 0.0


weight_decay = 1e-06


backtracking_weight_percentage = 0.0

patience = 10
plateau_cutoff = 1e-04
