# Glow Library
Collection of tools for easier prototyping with deep learning extensions (PyTorch framework)

## Overview
...

## Installation

If you don't need neural network or I/O features use common installation:

```bash
pip install -U glow
```
<details>
<summary>Specific versions with additional requirements</summary>

```bash
pip install -U glow[cv]  # If you need cv/neural network features
pip install -U glow[io]  # If you need io features
pip install -U glow[cv,io]  # If you need all
```
</details>
Glow is compatible with: Python 3.8+, PyTorch 1.7+.
Tested on ArchLinux, Ubuntu 18.04/20.04, Windows 10.

## Structure
- `glow.*` - core parts, available out the box
- `glow.io.*` - IO wrappers to access data in convenient formats
- `glow.transforms` - Some custom-made augmentations for data
- `glow.nn.` - Neural nets and building blocks for them
- `glow.metrics` - Metric to use while training your neural network

## Core features
- `glow.mapped` - convenient tool to parallelize computations
- `glow.memoize` - use if you want to reduce number of calls for any function

## IO features

### `glow.io.TiledImage` - ndarray-like reader for multiscale images (svs, tiff, etc...)
<details>

CTypes-based replacement of [`torchslide`](https://github.com/arquolo/torchslide) (deprecated).

```python
from glow.io import read_tiled

slide = read_tiled('test.svs')
shape: tuple[int, ...] = slide.shape
scales: tuple[int, ...] = slide.scales
image: np.ndarray = slide[:2048, :2048].view()  # Get numpy.ndarray
```
</details>

### `glow.io.Sound` - playable sound wrapper
<details>

```python
import numpy as np
from glow.io import Sound

array: np.ndarray
sound = Sound(array, rate=44100)  # Wrap np.ndarray
sound = Sound.load('test.flac')  # Load sound into memory from file

rate: int = sound.rate
shape: Tuple[int, int] = sound.shape
dtype: np.dtype = sound.dtype
sound.play()  # Plays sound through default device
```
</details>
