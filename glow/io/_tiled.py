__all__ = ['TiledImage', 'read_tiled']

import ctypes
import os
import sys
import weakref
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, NamedTuple, Optional, Tuple, Type, Union

import cv2
import numpy as np

from .. import call_once, memoize

_TIFF: Any = None
_OSD: Any = None
_TYPE_REGISTRY: Dict[str, Type['TiledImage']] = {}


def _patch_path(prefix):
    if sys.platform != 'win32':
        return nullcontext()
    return os.add_dll_directory(prefix)


@call_once
def _setup_libs():
    global _TIFF, _OSD
    prefix = Path(__file__).parent / 'libs'
    pattern = ((prefix / '{}-{}.dll').as_posix()
               if sys.platform == 'win32' else '{}.so.{}')
    names = map(pattern.format, ('libtiff', 'libopenslide'), (5, 0))

    with _patch_path(prefix):
        _TIFF, _OSD = map(ctypes.CDLL, names)

    struct_t = ctypes.POINTER(ctypes.c_ubyte)
    (_TIFF.TIFFOpenW
     if sys.platform == 'win32' else _TIFF.TIFFOpen).restype = struct_t

    _TIFF.TIFFSetErrorHandler(None)
    _OSD.openslide_open.restype = struct_t
    _OSD.openslide_get_error.restype = ctypes.c_char_p
    _OSD.openslide_get_property_value.restype = ctypes.c_char_p
    _OSD.openslide_get_property_names.restype = ctypes.POINTER(ctypes.c_char_p)


# ----------------------- some constants from libtiff -----------------------


class Color(Enum):
    MINISBLACK = 1
    RGB = 2
    YCBCR = 6


class Codec(Enum):
    RAW = 1
    CCITT = 2
    CCITTFAX3 = 3
    CCITTFAX4 = 4
    LZW = 5
    JPEG_OLD = 6
    JPEG = 7
    ADOBE_DEFLATE = 8
    RAW_16 = 32771
    PACKBITS = 32773
    THUNDERSCAN = 32809
    DEFLATE = 32946
    DCS = 32947
    JPEG2000_YUV = 33003
    JPEG2000_RGB = 33005
    JBIG = 34661
    SGILOG = 34676
    SGILOG24 = 34677
    JPEG2000 = 34712
    LZMA = 34925
    ZSTD = 50000
    WEBP = 50001


# --------------------------------- rescaler ---------------------------------


class _TileScaler(NamedTuple):
    """Proxy, moves rescaling out of `TiledImage.__getitem__`"""
    image: np.ndarray
    scale: float = 1.

    def view(self, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Converts region to array of desired size.
        If size is not set, rounding will be used,
        otherwise it should be `(height, width)`.
        """
        if size is None:
            if self.scale == 1.:
                return self.image

            h, w = self.image.shape[:2]
            size = round(self.scale * h), round(self.scale * w)

        return cv2.resize(self.image, size[::-1], interpolation=cv2.INTER_AREA)


# --------------------------------- decoders ---------------------------------


class _Decoder:
    def __init__(self):
        self._lock = RLock()

    def _get_spec(self, level: int) -> dict:
        raise NotImplementedError

    def _get_patch(self, box, **spec) -> np.ndarray:
        raise NotImplementedError

    @property
    def spacing(self) -> Optional[float]:
        raise NotImplementedError

    @contextmanager
    def _directory(self, level):
        with self._lock:
            yield


class TiledImage(_Decoder):
    def __init_subclass__(cls: Type['TiledImage'], extensions: str) -> None:
        _TYPE_REGISTRY.update({f'.{ext}': cls for ext in extensions.split()})

    def __init__(self, path: Path, num_levels: int = 0) -> None:
        if type(self) is TiledImage:
            raise RuntimeError('TiledImage is not for direct construction. '
                               'Use read_tiled() factory function')
        super().__init__()
        self.path = path
        self._num_levels = num_levels
        self._spec = dict(self._init_spec(num_levels))

    def __reduce__(self) -> Union[str, tuple]:
        return TiledImage, (self.path, )  # self._slices)

    def _init_spec(self, num_levels: int):
        assert num_levels > 0
        zero, *specs = (self._get_spec(level) for level in range(num_levels))
        shape = zero['shape']
        for spec in [zero, *specs]:
            if spec:
                spec['step'] = step = round(shape[0] / spec['shape'][0])
                yield step, spec

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._spec[1]['shape']

    @property
    def scales(self) -> Tuple[int, ...]:
        return tuple(self._spec.keys())

    def __repr__(self) -> str:
        return (f'{type(self).__name__}'
                f"('{self.path}', shape={self.shape}, scales={self.scales})")

    def __getitem__(self, slices: Union[Tuple[slice, slice],
                                        slice]) -> _TileScaler:
        """Retrieves tile"""
        if isinstance(slices, slice):
            slices = (slices, slice(None, None, slices.step))
        assert slices[0].step is None or slices[0].step >= 1, (
            'Step should be None or greater than 1')
        assert slices[0].step == slices[1].step, (
            'Unequal steps for Y and X are not supported')

        rstep = slices[0].step or 1
        step = max((s for s in self._spec if s <= rstep), default=1)
        spec = self._spec[step]

        box = [((0 if s.start is None else s.start // step),
                (lim if s.stop is None else s.stop // step))
               for s, lim in zip(slices, spec['shape'])]

        with self._directory(spec['level']):
            image = self._get_patch(box, **spec)
        return _TileScaler(image, step / rstep)

    def thumbnail(self, scale: int = None) -> Tuple[np.ndarray, int]:
        if scale is None:
            scale = self.scales[-1]
        return self[::scale, ::scale].view(), scale

    def at(self,
           z0_yx_offset: Tuple[int, int],
           dst_size: int,
           scale: int = 1) -> np.ndarray:
        y, x = z0_yx_offset
        return self[y:y + dst_size * scale:scale,
                    x:x + dst_size * scale:scale].view((dst_size, dst_size))


class _OpenslideImage(
        TiledImage,
        extensions='bif mrxs ndpi scn svs svsslide tif tiff vms vmu'):
    def __init__(self, path: Path) -> None:
        _setup_libs()
        self._ptr = _OSD.openslide_open(path.as_posix().encode())
        if err := _OSD.openslide_get_error(self._ptr):
            raise ValueError(err)
        weakref.finalize(self, _OSD.openslide_close, self._ptr)

        if (bg_color_hex := _OSD.openslide_get_property_value(
                self._ptr, b'openslide.background-color')) is not None:
            iv = int(bg_color_hex, base=16).to_bytes(3, 'big')
            self.bg_color = np.frombuffer(iv, dtype='u1')
        else:
            self.bg_color = np.full(3, 255, dtype='u1')

        num_levels = _OSD.openslide_get_level_count(self._ptr)
        super().__init__(path, num_levels)

    def _get_spec(self, level: int) -> dict:
        y, x = ctypes.c_int64(), ctypes.c_int64()
        _OSD.openslide_get_level_dimensions(self._ptr, level,
                                            *map(ctypes.byref, (x, y)))
        return {'level': level, 'shape': (y.value, x.value)}

    def _get_patch(self, box, step=0, level=0, **_):
        (y_min, y_max), (x_min, x_max) = box

        data = np.empty((y_max - y_min, x_max - x_min, 4), dtype='u1')
        data_ptr = ctypes.c_void_p(data.ctypes.data)
        _OSD.openslide_read_region(self._ptr, data_ptr, x_min * step,
                                   y_min * step, level, x_max - x_min,
                                   y_max - y_min)

        rgb = data[..., 2::-1]
        opacity = data[..., 3:]
        u2_255 = np.array(255, dtype='u2')
        return np.where(opacity, (u2_255 * rgb / opacity.clip(1)).astype('u1'),
                        self.bg_color)

    @property
    def metadata(self) -> Dict[str, str]:
        m: Dict[str, str] = {}
        if names := _OSD.openslide_get_property_names(self._ptr):
            i = 0
            while name := names[i]:
                m[name.decode()] = _OSD.openslide_get_property_value(
                    self._ptr, name).decode()
                i += 1
        return m

    @property
    def spacing(self) -> Optional[float]:
        mpp = (
            _OSD.openslide_get_property_value(
                self._ptr, f'openslide.mpp-{ax}'.encode()) for ax in 'yx')
        if s := [float(m) for m in mpp if m]:
            return np.mean(s)
        return None


class _TiffImage(TiledImage, extensions='svs tif tiff'):
    def __init__(self, path: Path) -> None:
        _setup_libs()
        self._ptr = (
            _TIFF.TIFFOpenW(path.as_posix(), b'rm') if sys.platform == 'win32'
            else _TIFF.TIFFOpen(path.as_posix().encode(), b'rm'))
        assert self._ptr
        weakref.finalize(self, _TIFF.TIFFClose, self._ptr)

        num_levels = _TIFF.TIFFNumberOfDirectories(self._ptr)
        super().__init__(path, num_levels)

    def _tag(self, type_, tag: int, default=None):
        value = type_()
        res = _TIFF.TIFFGetField(self._ptr, ctypes.c_uint32(tag),
                                 ctypes.byref(value))
        if not res and default is not None:
            return default
        return value.value

    @contextmanager
    def _directory(self, level):
        with super()._directory(level):
            _TIFF.TIFFSetDirectory(self._ptr, level)
            try:
                yield
            finally:
                _TIFF.TIFFFreeDirectory(self._ptr)

    def _get_spec(self, level) -> dict:
        _TIFF.TIFFSetDirectory(self._ptr, level)

        if not _TIFF.TIFFIsTiled(self._ptr):
            return {}

        if self._tag(ctypes.c_uint16, 284) != 1:
            raise TypeError(f'Level {level} is not contiguous!')

        desc = self._tag(ctypes.c_char_p, 270)
        desc = (desc or b'').decode().replace('\r\n', '|').split('|')

        color = Color(self._tag(ctypes.c_uint16, 262))
        spp = (
            self._tag(ctypes.c_uint16, 277)
            if color in [Color.MINISBLACK, Color.RGB] else 4)
        shape = (self._tag(ctypes.c_uint32, tag) for tag in (257, 256))
        tshape = (self._tag(ctypes.c_uint32, tag) for tag in (323, 322))

        spec = {
            'level': level,
            'shape': (*shape, spp),
            'tile': (*tshape, spp),
            'color': color,
            # 'bits_per_sample': self._tag(ctypes.c_uint16, 258),
            # 'sample_format': self._tag(ctypes.c_uint16, 339, default=1),
            'description': desc,
            'compression': Codec(self._tag(ctypes.c_uint16, 259)),
        }
        if spec['compression'] is Codec.JPEG:
            count = ctypes.c_int()
            jpt_ptr = ctypes.c_char_p()
            if _TIFF.TIFFGetField(
                    self._ptr, 347, ctypes.byref(count),
                    ctypes.byref(jpt_ptr)) and count.value > 4:
                spec['jpt'] = ctypes.string_at(jpt_ptr, count.value)

        return spec

    def _get_tile(self, y, x, **spec) -> np.ndarray:
        if spec['compression'] not in [Codec.JPEG2000_RGB, Codec.JPEG2000_YUV]:
            image = np.empty(spec['tile'], dtype='u1')
            isok = _TIFF.TIFFReadTile(self._ptr,
                                      ctypes.c_void_p(image.ctypes.data), x, y,
                                      0, 0)
            assert isok != -1
            return image

        offset = _TIFF.TIFFComputeTile(self._ptr, x, y, 0, 0)
        tile_byte_counts = self._tag(ctypes.c_void_p, 325)
        nbytes: int = ctypes.cast(tile_byte_counts,
                                  ctypes.POINTER(ctypes.c_ulonglong))[offset]

        data = ctypes.create_string_buffer(nbytes)
        _TIFF.TIFFReadRawTile(self._ptr, offset, data, len(data))

        import imagecodecs
        if jpt := spec.get('jpt'):
            return imagecodecs.jpeg_decode(
                data, tables=jpt, colorspace=spec['color'].value)
        return imagecodecs.imread(data)

    def _get_patch(self, box, **spec) -> np.ndarray:
        *tile, spp = spec['tile']

        dy, dx = (low for low, _ in box)
        out = np.zeros([(high - low) for low, high in box] + [spp], dtype='u1')

        bmin, bmax = np.transpose(box).clip(0, spec['shape'][:2])
        axes = *map(slice, bmin // tile * tile, bmax, tile),
        grid = np.mgrid[axes].transpose(1, 2, 0)  # [H, W, 2]
        if not grid.size:
            return out

        # * block, a bit slower
        # patches = np.block([[[self._get_tile(*iyx, **spec)] for iyx in iyxs]
        #                     for iyxs in grid.tolist()])
        # dst = *map(slice, bmin - (dy, dx), bmax - (dy, dx)),
        # src = *map(slice, bmin % tile, patches.shape[:2] - (-bmax) % tile),
        # out[dst] = patches[src]
        # return out

        # * iterate
        grid = grid.reshape(-1, 2)
        for (iy, ix), (ty_min, tx_min), (ty_max, tx_max) in zip(
                grid.tolist(),
                grid.clip(bmin).tolist(),
                np.clip(grid + tile, 0, bmax).tolist()):
            patch = self._get_tile(iy, ix, **spec)
            out[ty_min - dy:ty_max - dy,
                tx_min - dx:tx_max - dx] = patch[ty_min - iy:ty_max - iy,
                                                 tx_min - ix:tx_max - ix]
        return out

    @property
    def spacing(self) -> Optional[float]:
        s = [(10000 / m)
             for x in (283, 282) if (m := self._tag(ctypes.c_float, x))]
        if s:
            return np.mean(s)
        for token in self._spec[1]['description']:
            if 'MPP' in token:
                return float(token.split('=')[-1].strip())
        return None


@memoize(
    10_485_760,
    policy='lru',
    key_fn=lambda name: Path(name).resolve().as_posix())
def read_tiled(anypath: Union[Path, str]) -> TiledImage:
    """Reads multi-scale images.

    Usage:
    ```
    from glow.io import read_tiled

    slide = read_tiled('test.svs')
    shape: tuple[int, ...] = slide.shape
    scales: tuple[int, ...] = slide.scales

    # Get numpy.ndarray
    image = slide[:2048, :2048].view()
    ```
    """
    if (path := Path(anypath)).exists():
        try:
            return _TYPE_REGISTRY[path.suffix](path)
        except KeyError:
            raise ValueError(f'Unknown file format {path}') from None
    raise FileNotFoundError(path)
