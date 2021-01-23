from __future__ import annotations  # until 3.10

__all__ = [
    'bit_noise', 'cutout', 'dither', 'elastic', 'grid_shuffle', 'hsv', 'jpeg',
    'multi_noise', 'Compose', 'Transform'
]

from types import MappingProxyType
from typing import Literal, Protocol

import cv2
import numba
import numpy as np

_MATRICES = MappingProxyType({
    key: cv2.normalize(np.float32(mat), None, norm_type=cv2.NORM_L1)
    for key, mat in {
        'jarvis-judice-ninke': [
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1],
        ],
        'sierra': [[0, 0, 0, 5, 3], [2, 4, 5, 4, 2], [0, 2, 3, 2, 0]],
        'stucki': [[0, 0, 0, 8, 4], [2, 4, 8, 4, 2], [1, 2, 4, 2, 1]],
    }.items()
})
_DitherKind = Literal['jarvis-judice-ninke', 'stucki', 'sierra']


class Transform(Protocol):
    def __call__(self, *args: np.ndarray,
                 rg: np.random.Generator) -> tuple[np.ndarray, ...]:
        ...


class Compose:
    def __init__(self, *callables: tuple[float, Transform]):
        self.callables = callables

    def __call__(self, *args: np.ndarray,
                 rg: np.random.Generator) -> tuple[np.ndarray, ...]:
        for prob, fn in self.callables:
            if rg.uniform() <= prob:
                new_args = fn(*args, rg=rg)
                args = new_args + args[len(new_args):]

        return args


@numba.jit
def _dither(image, mat, quant):
    for y in range(image.shape[0] - 2):
        for x in range(2, image.shape[1] - 2):
            old = image[y, x]
            new = np.floor(old / quant) * quant
            delta = ((old - new) * mat).astype(image.dtype)
            image[y:y + 3, x - 2:x + 3] += delta
            image[y, x] = new

    return image[:-2, 2:-2]


def dither(image: np.ndarray,
           *_,
           rg: np.random.Generator = None,
           bits: int = 3,
           kind: _DitherKind = 'stucki') -> tuple[np.ndarray, ...]:
    mat = _MATRICES[kind]
    assert image.dtype == 'u1'
    if image.ndim == 3:
        mat = mat[..., None]
        channel_pad = [(0, 0)]
    else:
        channel_pad = []
    image = np.pad(image, [(0, 2), (2, 2)] + channel_pad, mode='constant')

    dtype = image.dtype
    if dtype == 'uint8':
        max_value = 256
        image = image.astype('i2')
    else:
        max_value = 1
    quant = max_value / 2 ** bits

    image = _dither(image, mat, quant)
    return image.clip(0, max_value - quant).astype(dtype),


def bit_noise(image: np.ndarray,
              *_,
              rg: np.random.Generator,
              keep: int = 4,
              count: int = 8) -> tuple[np.ndarray]:
    residual = image.copy()
    out = np.zeros_like(image)
    for n in range(1, 1 + count):
        thres = 0.5 ** n
        plane = (residual >= thres).astype(residual.dtype) * thres
        if n <= keep:
            out += plane
        else:
            out += rg.choice([0, thres], size=image.shape)
        residual -= plane
    return out,


def elastic(image: np.ndarray,
            mask: np.ndarray = None,
            *,
            rg: np.random.Generator,
            scale: float = 1,
            sigma: float = 50,
            interp=cv2.INTER_LINEAR,
            border=cv2.BORDER_REFLECT_101) -> tuple[np.ndarray, ...]:
    """Elastic deformation of image

    Parameters:
    - scale - max offset for each pixel
    - sigma - size of gaussian kernel
    """
    offsets = rg.random((2, *image.shape[:2]), dtype='f4')
    offsets *= (2 * scale)
    offsets -= scale

    for dim, (off, size) in enumerate(zip(offsets, image.shape[:2])):
        shape = np.where(np.arange(2) == dim, size, 1)
        off += np.arange(size).reshape(shape)
        cv2.GaussianBlur(off, (17, 17), sigma, dst=off)
    offsets = offsets[::-1]

    image = cv2.remap(image, *offsets, interp, borderMode=border)
    if mask is None:
        return image,
    mask = cv2.remap(mask, *offsets, cv2.INTER_NEAREST, borderMode=border)
    return image, mask


def cutout(image: np.ndarray,
           *_,
           rg: np.random.Generator,
           max_holes=80,
           size=8,
           fill_value=0) -> tuple[np.ndarray]:
    num_holes = rg.integers(max_holes)
    if not num_holes:
        return image,

    anchors = rg.integers(0, image.shape[:2], size=(num_holes, 2))

    # [N, dims, (min, max)]
    holes = anchors[:, :, None] + [-size // 2, size // 2]
    holes = holes.clip(0, np.array(image.shape[:2])[:, None])

    image = image.copy()
    for (y0, y1), (x0, x1) in holes:
        image[y0:y1, x0:x1] = fill_value
    return image,


def jpeg(image: np.ndarray,
         *_,
         rg: np.random.Generator,
         low=0,
         high=15) -> tuple[np.ndarray]:
    assert image.dtype == np.uint8
    quality = int(rg.integers(low=low, high=high))
    _, buf = cv2.imencode('.jpg', image, (cv2.IMWRITE_JPEG_QUALITY, quality))
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).reshape(image.shape),


def multi_noise(image: np.ndarray,
                *_,
                rg: np.random.Generator,
                low=0.5,
                high=1.5,
                elementwise=False) -> tuple[np.ndarray]:
    assert image.dtype == np.uint8
    if elementwise:
        mask = rg.random(image.shape[:2], dtype='f4')
        mask *= (high - low)
        mask -= low
        return (image * mask[..., None]).clip(0, 255).astype('u1'),

    lut = np.arange(0, 256, dtype=np.float32)
    lut *= rg.uniform(low=low, high=high)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    image = cv2.merge(
        [cv2.LUT(plane, lut) for plane in cv2.split(image)]).reshape(
            image.shape)

    return image,


def hsv(image: np.ndarray,
        *_,
        rg: np.random.Generator,
        max_shift=20) -> tuple[np.ndarray]:
    assert image.dtype == np.uint8
    assert image.ndim == image.shape[-1] == 3
    hue, sat, val = rg.uniform(-max_shift, max_shift, size=3)
    lut = np.arange(256, dtype=np.int16)
    luts = [
        (lut + hue) % 180,
        (lut + sat).clip(0, 255),
        (lut + val).clip(0, 255),
    ]

    shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image = cv2.merge([
        cv2.LUT(plane, lut.astype('u1'))
        for lut, plane in zip(luts, cv2.split(image))
    ])

    image = cv2.cvtColor(image.astype('u1'), cv2.COLOR_HSV2RGB)
    return image.reshape(shape),


def grid_shuffle(image: np.ndarray,
                 mask: np.ndarray = None,
                 *,
                 rg: np.random.Generator,
                 grid=4) -> tuple[np.ndarray, ...]:
    axes = (
        np.linspace(0, size, grid + 1, dtype=np.int)
        for size in image.shape[:2])
    mats = np.stack(np.meshgrid(*axes, indexing='ij'))

    anchors = mats[:, :-1, :-1]
    tiles_sizes = (mats[:, 1:, 1:] - anchors).transpose(1, 2, 0)
    indices = np.stack(np.indices((grid, grid)), axis=2)

    for axis in range(2):
        indices = rg.permutation(indices, axis=axis)

    for bbox_size in np.unique(tiles_sizes.reshape(-1, 2), axis=0):
        eq_mat = np.all(tiles_sizes == bbox_size, axis=2)
        indices[eq_mat] = rg.permutation(indices[eq_mat])

    old_pos = anchors[:, indices[..., 0], indices[..., 1]].reshape(2, -1)
    new_pos = anchors.reshape(2, -1)

    sizes = tiles_sizes.reshape(-1, 2)
    tiles = np.stack([new_pos.T, old_pos.T, sizes], axis=1)

    sources = [image]
    if mask is not None:
        sources.append(mask)

    results = []
    for v in sources:
        new_v = np.empty_like(v)
        for (y2, x2), (y1, x1), (ys, xs) in tiles:
            new_v[y2:y2 + ys, x2:x2 + xs] = v[y1:y1 + ys, x1:x1 + xs]
        results.append(new_v)
    return *results,
