# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os


data_path = "{0}/DIP_RDP_data/static_mean_thorax_simulation_noisy/".format(os.path.dirname(os.getcwd()))
output_path = "{0}/output/static_mean_thorax_simulation_noisy/".format(os.getcwd())

gaussian_path = "{0}/input/untrained/".format(os.getcwd())

data_window_size = 64
data_window_bool = True

data_resample_power_of = 2

data_input_bool = True

data_gaussian_smooth_sigma_xy = 0.0
data_gaussian_smooth_sigma_z = 0.0

input_gaussian_weight = 1.0


model_path = "{0}/input/untrained/".format(os.getcwd())

new_model_patient_bool = False
new_optimiser_patient_bool = True

new_model_window_bool = False
new_optimiser_window_bool = True


input_gaussian_sigma = 0.0
skip_gaussian_sigma = 0.0
layer_gaussian_sigma = 0.0


dropout = 0.0

bayesian_test_bool = False
bayesian_output_bool = False
bayesian_iterations = 1


log_cosh_weight = 1e00

relative_difference_bool = True
relative_difference_weight = 1e-01
relative_difference_edge_preservation_weight = 1e-03

l1_weight_activity = 0.0
l1_weight_prelu = 0.0
l2_weight = 0.0

uncertainty_weight = 0.0


weight_decay = 0.0


backtracking_weight_percentage = 0.0

patience = 10
plateau_cutoff = 1e-04
