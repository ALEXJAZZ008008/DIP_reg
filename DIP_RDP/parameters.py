# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


data_window_size = 47
data_window_bool = False

data_resample_power_of = 2


data_input_bool = True

data_gaussian_smooth_sigma_xy = 1.5
data_gaussian_smooth_sigma_z = 1.0

input_gaussian_weight = 0.3


input_skip_gaussian_sigma = 0.3
layer_gaussian_sigma = 0.01

bayesian_bool = True
bayesian_test_iterations = 8

dropout = 0.1
uncertainty_weight = 0.0


l1_weight = 1e-02
l2_weight = 1e-00

total_variation_bool = True
total_variation_weight = 1e-00

relative_difference_bool = False
relative_difference_weight = 0.0
relative_difference_edge_preservation_weight = 0.0


weight_decay = 0.0


new_model_patient_bool = False
new_model_window_bool = False


backtracking_weight_percentage = 0.0


patience = 10
plateau_cutoff = 1e-04
