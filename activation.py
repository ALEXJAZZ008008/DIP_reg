# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow.keras as k


def leaky_relu_6(x, leaky_alpha=0.2, six_alpha=0.04):
    return k.backend.minimum(k.backend.maximum(x,
                                               k.backend.maximum(x * leaky_alpha,
                                                                 -6 - (x * six_alpha))),
                             6 + (x * six_alpha))
