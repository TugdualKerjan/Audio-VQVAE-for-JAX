# Audio-VQVAE-for-JAX

[![Project Status: Working](https://img.shields.io/badge/status-working-brightgreen.svg)](https://github.com/yourusername/Audio-VQVAE-for-JAX)

A JAX-based implementation of Vector-Quantized Variational Autoencoder (VQ-VAE) tailored for audio processing. This model is part of the JAXTTS (eXtended Text-To-Speech) series, where I rewrite XTTS in JAX to understand how it works from A to Z, and learn JAX along the way.

## Overview

This project leverages **JAX** and **Equinox** for a VQ-VAE model focused on audio data, copying [XTTS](https://github.com/coqui-ai/TTS)'s structure.

## Features

- Encoder-Decoder architecture for audio data using Conv1d and UpsampleConv1d.
- Comprehensive JAX and Equinox integration
- Documentation with step-by-step tutorials and explanations for each module

## Getting Started

To get started, clone the repository and install the necessary dependencies:

```bash
git clone git@github.com:TugdualKerjan/Audio-VQVAE-for-JAX.git
cd Audio-VQVAE-for-JAX
pip install -r requirements.txt