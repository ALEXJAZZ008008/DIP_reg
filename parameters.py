# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


noise_input = True


autoencoder_resnet_bool = False
autoencoder_unet_bool = True
autoencoder_unet_concatenate_bool = False

down_stride_bool = True
down_max_pool_too_bool = False
up_stride_bool = False
up_upsample_too_bool = False


transform_bool = False
elastic_sigma = 0.0

noise_scale = 0.0
background_scale = 0.0


grouped_bool = False
grouped_channel_shuffle_bool = False

gaussian_stddev = 0.0
lone_weight = 0.0

dropout_amount = 0.0


bottleneck_expand_multiplier = 8


depthwise_seperable_bool = False
bottleneck_bool = False
