# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


data_window_size = 47
data_window_bool = True

data_resample_power_of = 2


data_input_bool = True

data_gaussian_smooth_sigma_xy = 0.5
data_gaussian_smooth_sigma_z = 0.5

input_gaussian_weight = 0.2


input_skip_gaussian_sigma = 0.0
layer_gaussian_sigma = 0.0

bayesian_test_bool = False
bayesian_output_bool = True
bayesian_iterations = 8

dropout = 0.1
uncertainty_weight = 0.0


l1_weight_activity = 1e-00
l1_weight_prelu = 1e01
l2_weight = 1e-03

total_variation_bool = False
total_variation_weight = 0.0

relative_difference_bool = True
relative_difference_weight = 1e-02
relative_difference_edge_preservation_weight = 1e01

max_min_constraint_weight = 1e-03


weight_decay = 1e-02


new_model_patient_bool = False
new_model_window_bool = False


backtracking_weight_percentage = 0.0


relative_plateau_cutoff_bool = False
patience = 10
plateau_cutoff = 1e-04
