#!/usr/bin/env python3
"""Task 0"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    Going Deeper with Convolutions (2014)
    :param A_prev: output from previous layer
    :param filters: tuple or list containing
    F1, F3R, F3,F5R, F5, FPP, respectively:
        F1: number of filters in the 1x1 convolution
        F3R: number of filters in 1x1 convolution before 3x3 convolution
        F3: number of filters in the 3x3 convolution
        F5R: number of filters in 1x1 convolution before 5x5 convolution
        F5: number of filters in the 5x5 convolution
        FPP: number of filters in 1x1 convolution after max pooling
    :return: concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(
        filters=F1, kernel_size=(1, 1),
        padding='same', activation='relu')(A_prev)

    conv3r = K.layers.Conv2D(
        filters=F3R, kernel_size=(1, 1),
        padding='same', activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3),
        padding='same', activation='relu')(conv3r)

    conv5r = K.layers.Conv2D(
        filters=F5R, kernel_size=(1, 1),
        padding='same', activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(
        filters=F5, kernel_size=(5, 5),
        padding='same', activation='relu')(conv5r)

    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1),
        padding='same')(A_prev)
    pool_conv = K.layers.Conv2D(
        filters=FPP, kernel_size=(1, 1),
        padding='same', activation='relu')(pool)

    output = K.layers.concatenate(
        [conv1, conv3, conv5, pool_conv], axis=3)

    return output
