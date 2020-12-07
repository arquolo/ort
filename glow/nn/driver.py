__all__ = ['get_gpu_state']

import os
from contextlib import ExitStack
from typing import NamedTuple, Sequence

from .. import Si


class _GpuState(NamedTuple):
    num_devices: int
    free: Si
    used: Si
    total: Si


def get_gpu_state() -> _GpuState:
    """Returns count of available GPUs and size of free memory VRAM"""
    from py3nvml.py3nvml import (nvmlDeviceGetCount,
                                 nvmlDeviceGetHandleByIndex,
                                 nvmlDeviceGetMemoryInfo, nvmlInit,
                                 nvmlShutdown)
    with ExitStack() as stack:
        nvmlInit()
        stack.callback(nvmlShutdown)

        indices: Sequence[int]
        devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if devices is not None:
            indices = [int(dev) for dev in devices.split(',')]
        else:
            indices = range(int(nvmlDeviceGetCount()))

        handles = (nvmlDeviceGetHandleByIndex(i) for i in indices)
        infos = (nvmlDeviceGetMemoryInfo(h) for h in handles)
        stats = [(i.free, i.used, i.total) for i in infos]

    return _GpuState(len(indices), *(Si.bits(sum(s)) for s in zip(*stats)))
