# pip install tensorflow_addons
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
# import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import h5py
import numpy as np
import math
import os
import random
# from __future__ import division
from scipy import linalg

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array

from .Unet_utils import kernel_init

from .Unet_utils import TimeEmbedding,TimeMLP,ResidualBlock,DownSample,UpSample,AttentionBlock
# Build a Unet model
# Build a Unet model
def build_model(
        input,
        time_input,
        label_input,
        widths,
        has_attention,
        first_conv_channels,
        num_res_blocks=2,
        norm_groups=8,
        interpolation="nearest",
        activation_fn=keras.activations.swish,
):
    # image_input = input
    image_input = tf.concat((input, label_input), axis=1)
    image_input = tf.reshape(image_input, shape=(-1,image_input.shape[1], 1))
    x = layers.ZeroPadding1D(padding=1)(image_input) #显示填0
    # x = tf.reshape(image_input, shape=(-1, image_input.shape[1], 1))
    x = layers.Conv1D(
        first_conv_channels,
        kernel_size=5,
        strides=1,
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(x)  # (None,64,64,64)

    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)  # (None,256)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)  # (None,256)
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])

            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)
        print('down block',i+1,'shape is', x.shape)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    print('middle block', 1, 'shape is', x.shape)
    # x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    print('middle block', 2, 'shape is', x.shape)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    print('middle block',3,'shape is', x.shape) #none 17 8


    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i])(x)
        print('up block', i+1, 'shape is', x.shape)
    # End block
    x = tfa.layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    # x = layers.UpSampling1D(size= 2)(x)
    print('end block', 1, 'shape is', x.shape)
    x = layers.Conv1D(1, 5, padding="same", kernel_initializer=kernel_init(0.0))(x)
    print('end block', 2, 'shape is', x.shape)
    x=tf.squeeze(x)
    return x




