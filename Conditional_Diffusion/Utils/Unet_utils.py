# pip install tensorflow_addons
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import math
import tensorflow_addons as tfa

#NN
# Unet Operate
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=1, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        # self.norm = tf.keras.layers.experimental.preprocessing.Normalization

        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        # print(proj)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[2]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv1D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, :]

        x = tfa.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv1D(
            width, kernel_size=5, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        # print(x.shape,temb.shape)

        x = layers.Add()([x, temb])
        # print(x.shape)
        x = tfa.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv1D(
            width, kernel_size=5, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        # print(x)
        x = layers.Add()([x, residual])

        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv1D(
            width,
            kernel_size=3,
            strides= 2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width):
    def apply(x):
        x = layers.UpSampling1D(size= 2)(x)
        x = layers.Conv1D(width, kernel_size=5, padding="same",activation=tf.nn.relu)(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply
