from functools import partial

import jax
import jax.numpy as jnp
import librosa
import numpy as np
from jax import lax
from jaxtyping import Array, Float

# Pre-compute static values
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
WINDOW = jnp.hanning(WIN_LENGTH)
PAD_SIZE = int((N_FFT - HOP_LENGTH) / 2)


def mel_spec_base(wav: Float[Array, "audio"]) -> Float[Array, "num_mel time"]:
    """Mel transform, takes in 22050 Hz signal

    Args:
        wav (jax.Array): 1D tensor

    Returns:
        jax.Array: (80, N) tensor
    """
    wav = np.pad(
        wav,
        ((0, 0), (PAD_SIZE, PAD_SIZE)),
        mode="reflect",
    )
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        power=2,
        window="hann",
        n_mels=N_MELS,
        pad_mode="reflect",
        fmax=8000,
        center=False,
        fmin=0,
    )
    mel = jnp.squeeze(mel, 0)
    mel = jnp.log(jnp.clip(mel, min=1e-5))  # Spectral normalization
    return mel


# Pre-compute mel filterbank (this is static)
def create_mel_filterbank():
    f_min = 0.0
    f_max = 8000.0
    n_freqs = (N_FFT // 2) + 1

    # Mel points calculation
    m_min = 2595.0 * jnp.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * jnp.log10(1.0 + (f_max / 700.0))
    m_pts = jnp.linspace(m_min, m_max, N_MELS + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)

    # Create filterbank
    all_freqs = jnp.linspace(0, SAMPLE_RATE // 2, n_freqs)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[jnp.newaxis, ...] - all_freqs[..., jnp.newaxis]
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = jnp.maximum(0.0, jnp.minimum(down_slopes, up_slopes))

    return fb


# Pre-compute this once
MEL_FILTERBANK = create_mel_filterbank()


@partial(jax.jit, static_argnames=("n_fft", "hop_length"))
def stft_computation(wav, n_fft, hop_length):
    """Combined STFT computation"""
    if wav.ndim == 1:
        wav = wav.reshape(1, -1, 1)
    elif wav.ndim == 2:
        wav = wav[..., None]

    # Extract frames
    frames = lax.conv_general_dilated_patches(
        lhs=wav,
        filter_shape=(n_fft,),
        window_strides=(hop_length,),
        padding="VALID",
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 2, 1), rhs_spec=(2, 1, 0), out_spec=(0, 2, 1)
        ),
    )

    # Apply window and compute FFT
    frames = frames * WINDOW[None, None, :]
    output = jnp.fft.fft(frames, n=n_fft, axis=-1)
    return output[..., : (n_fft // 2) + 1]


@jax.jit
def mel_spec_base_jit(wav: Float[Array, "audio"]) -> Float[Array, "num_mel time"]:
    """Mel transform, takes in 22050 Hz signal"""
    # Pad the input
    wav = jnp.pad(wav, ((0, 0), (PAD_SIZE, PAD_SIZE)), mode="reflect")

    # Compute STFT
    spec = stft_computation(wav, N_FFT, HOP_LENGTH)

    # Convert to power spectrum
    spec_power = jnp.sqrt(jnp.real(spec) ** 2 + jnp.imag(spec) ** 2 + 1e-9)

    # Apply mel filterbank and log transform
    mel = jnp.matmul(spec_power, MEL_FILTERBANK, precision="highest")
    mel = jnp.log(jnp.clip(mel, min=1e-5))

    # Final shape adjustments
    mel = jnp.squeeze(mel, 0)
    mel = jnp.swapaxes(mel, axis1=0, axis2=1)
    return mel