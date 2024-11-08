#!/usr/bin/env python
# coding: utf-8



import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import typing as tp




class ResBlock(eqx.Module):
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d
    act: tp.Callable = eqx.static_field()

    def __init__(self, dim: int, activation=jax.nn.relu, key=None):

        key1, key2, key3 = jax.random.split(key, 3)

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=(1,), key=key1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=(1,), key=key2)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=1, key=key3)

        self.act = activation

    def __call__(self, x):
        y = x

        y = self.conv1(y)
        y = jax.nn.relu(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.conv3(y)

        y = y + x

        return y




class Encoder(eqx.Module):
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d
    res1: ResBlock
    res2: ResBlock
    res3: ResBlock

    def __init__(self, hidden_dim: int = 1024, codebook_dim: int = 512, key=None):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.conv1 = nn.Conv1d(
            in_channels=80,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=512,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key2,
        )
        self.res1 = ResBlock(dim=hidden_dim, key=key3)
        self.res2 = ResBlock(dim=hidden_dim, key=key4)
        self.res3 = ResBlock(dim=hidden_dim, key=key5)
        self.conv3 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=codebook_dim,
            kernel_size=1,
            stride=1,
            # padding="SAME",
            key=key6,
        )

    def __call__(self, x):

        y = self.conv1(x)
        y = jax.nn.relu(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.conv3(y)

        return y




class Decoder(eqx.Module):
    conv1: nn.Conv1d
    conv2: UpsampledConv
    conv3: UpsampledConv
    conv4: nn.Conv1d
    res1: ResBlock
    res2: ResBlock
    res3: ResBlock

    def __init__(self, hidden_dim: int = 1024, codebook_dim: int = 512, key=None):
        key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)

        self.conv1 = nn.Conv1d(
            in_channels=codebook_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            # padding="SAME",
            key=key1,
        )
        self.res1 = ResBlock(dim=hidden_dim, key=key2)
        self.res2 = ResBlock(dim=hidden_dim, key=key3)
        self.res3 = ResBlock(dim=hidden_dim, key=key4)
        self.conv2 = UpsampledConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key5,
        )
        self.conv3 = UpsampledConv(
            in_channels=hidden_dim,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key6,
        )
        self.conv4 = nn.Conv1d(
            in_channels=512,
            out_channels=80,
            kernel_size=1,
            stride=1,
            # padding="SAME",
            key=key7,
        )

    def __call__(self, x):

        y = self.conv1(x)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.conv3(y)
        y = jax.nn.relu(y)
        y = self.conv4(y)

        return y




# | label: upsample


class UpsampledConv(eqx.Module):
    conv: nn.Conv1d
    stride: int = eqx.static_field()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tp.Union[int, tp.Tuple[int]],
        stride: int,
        padding: tp.Union[int, str],
        key=None,
    ):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            key=key,
        )

    def __call__(self, x):
        upsampled_size = (x.shape[0], x.shape[1] * self.stride)
        upsampled = jax.image.resize(x, upsampled_size, method="nearest")
        return self.conv(upsampled)




# | output: true


class Quantizer(eqx.Module):
    K: int = eqx.static_field()
    D: int = eqx.static_field()
    codebook: jax.Array

    codebook_avg: jax.Array
    cluster_size: jax.Array

    decay: float = eqx.static_field()
    eps: float = eqx.static_field()

    def __init__(
        self,
        num_vecs: int = 1024,
        num_dims: int = 512,
        decay: float = 0.99,
        eps: float = 1e-5,
        key=None,
    ):
        self.K = num_vecs
        self.D = num_dims

        self.decay = decay
        self.eps = eps

        # Init a matrix of vectors that will move with time
        self.codebook = jax.nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="uniform"
        )(key, (num_vecs, num_dims))
        self.codebook_avg = jnp.copy(self.codebook)
        self.cluster_size = jnp.zeros(num_vecs)

    def __call__(self, x):
        # x has N vectors of the codebook dimension. We calculate the nearest neighbors and output those instead

        flatten = jax.numpy.reshape(x, (-1, self.D))
        a_squared = jnp.sum(flatten**2, axis=-1, keepdims=True)
        b_squared = jnp.transpose(jnp.sum(self.codebook**2, axis=-1, keepdims=True))
        distance = (
            a_squared
            + b_squared
            - 2 * jnp.matmul(flatten, jnp.transpose(self.codebook))
        )

        codebook_indices = jnp.argmin(distance, axis=-1)

        z_q = self.codebook[codebook_indices]

        # Straight-through estimator
        z_q = flatten + jax.lax.stop_gradient(z_q - flatten)

        # z_q = jax.numpy.permute_dims(z_q, (1, 0))
        z_q = jax.numpy.reshape(z_q, (-1, x.shape[-1]))

        return z_q, self.codebook_updates(flatten, codebook_indices)

    def codebook_updates(self, flatten, codebook_indices):

        # Calculate the usage of various codes.
        codebook_onehot = jax.nn.one_hot(codebook_indices, self.K)
        codebook_onehot_sum = jnp.sum(codebook_onehot, axis=0)
        codebook_sum = jnp.dot(flatten.T, codebook_onehot)
        # We've just weighed the codebook vectors.

        # Basically count on average how many codes we're using
        new_cluster_size = (
            self.decay * self.cluster_size + (1 - self.decay) * codebook_onehot_sum
        )

        # Where is the average embedding at ?
        new_codebook_avg = (
            self.decay * self.codebook_avg + (1 - self.decay) * codebook_sum.T
        )

        n = jnp.sum(new_cluster_size)  # Over the total embeddings used
        new_cluster_size = (new_cluster_size + self.eps) / (n + self.K * self.eps) * n
        new_codebook = self.codebook_avg / new_cluster_size[:, None]

        updates = (new_cluster_size, new_codebook_avg, new_codebook)

        return updates, codebook_indices




class VQVAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    quantizer: Quantizer

    def __init__(self, key=None):
        key1, key2, key3 = jax.random.split(key, 3)

        self.encoder = Encoder(key=key1)
        self.decoder = Decoder(key=key2)
        self.quantizer = Quantizer(decay=0.8, key=key3)

    def __call__(self, x):
        z_e = self.encoder(x)
        z_q, codebook_indices = self.quantizer(z_e)
        y = self.decoder(z_q)

        return z_e, z_q, codebook_indices, y
