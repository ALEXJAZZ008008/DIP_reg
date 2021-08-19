# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.

train_bool = True

autoencoder_bool = True
classifier_bool = True

cnn_only_autoencoder_bool = True
dv_bool = False

# suggested values
# follow for both from left to right for pseudo simulated annealing
# follow in inverse directions for traditional
# 1e-01 * (n * 0.1):    1e-01,  1e-02,  1e-03,  1e-04,  1e-05,  1e-06   etc
initial_learning_rate = 1e-02
# if using 4d data then each sample can be considered a batch for time distributed layers
# rounded down to nearest power of two
# 1 * (n * (1 / 0.1)):  1,      8,      64,     512,    8192,   65536   etc
initial_batch_size = 1

autoencoder_resnet_bool = True
autoencoder_unet_bool = True

down_stride_bool = True
down_max_pool_too_bool = True
up_stride_bool = True
up_upsample_too_bool = True


transform_bool = True
elastic_sigma = 0.0

noise_scale = 0.1
background_scale = 0.0


grouped_bool = True
grouped_channel_shuffle_bool = True

gaussian_stddev = 0.1
lone_weight = 0.0

# use 0.5 when training for real
dropout_amount = 0.0
beta = 1.0

latent_size = 8


bottleneck_expand_multiplier = 8


depthwise_seperable_bool = True
bottleneck_bool = True
