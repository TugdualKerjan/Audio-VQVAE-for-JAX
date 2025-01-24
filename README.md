# Audio-VAEs-JAX

[![Project Status: Working](https://img.shields.io/badge/status-working-brightgreen.svg)](https://github.com/bla)

A JAX-based implementation of Variational Autoencoders (VAE) and Vector-Quantized VAEs (VQ-VAE) tailored for audio processing. This model is part of the JAXTTS (eXtended Text-To-Speech) series, where I rewrite XTTS in JAX to understand how it works from A to Z, and learn JAX along the way.

## Overview

This project leverages **JAX** and **Equinox** to build a VAE / VQ-VAE model focused on audio data.

## 🚙 Roadmap

- [x] Have a functioning VAE
- [x] Provide checkpoints for the models
- [x] JIT Mel transform
- [x] Documentation with step-by-step tutorials and explanations for each module
- [x] Acceleration of the model using JIT
- [x] Logging of mel spectrograms
- [ ] Full type annotation
- [ ] Validation loss
- [ ] DocStrings for the functions
- [ ] Speed comparison with equivalent torch models

## Getting Started

To get started, follow the commands below. I recommend you use UV as a package manager:

```bash
git clone git@github.com:TugdualKerjan/audio-vae-jax.git
cd audio-vae-jax
uv sync
uv add jax["cuda"] # JAX has various versions optimized for the underlying architecture
uv run train_vae.py
```