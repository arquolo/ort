from __future__ import annotations  # until 3.10

__all__ = ['memprof', 'time_this', 'timer']

import functools
import threading
import time
import weakref
import inspect
from contextlib import contextmanager
from collections.abc import Callable
from typing import TypeVar, cast

from ._repr import Si

_F = TypeVar('_F', bound=Callable)
_THIS = None


@contextmanager
def memprof(name: str = 'Task', callback: Callable[[float], object] = None):
    global _THIS
    if _THIS is None:
        import psutil
        _THIS = psutil.Process()

    init = _THIS.memory_info().rss
    try:
        yield
    finally:
        size = _THIS.memory_info().rss - init
        if callback is not None:
            callback(size)
        else:
            print(f'{name} done: {"+" if size >= 0 else ""}{Si.bits(size)}')


@contextmanager
def timer(name: str = None, callback: Callable[[float], object] = None):
    init = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - init
        if callback is not None:
            callback(duration)
        else:
            if name is None:
                if ((frame := inspect.currentframe()) and frame.f_back and
                        frame.f_back.f_back):
                    tb = inspect.getframeinfo(frame.f_back.f_back)
                    name = f'{tb.filename}:{tb.lineno} block'
                else:
                    name = 'Task'

            print(f'{name} done in {Si(duration)}s')


def time_this(fn: _F) -> _F:
    """Log function timings at program exit"""
    infos: dict[int, list] = {}

    def finalize(start):
        if not infos:
            return

        num_calls = sum(num for num, _ in infos.values())
        total_time = sum(num * duration for num, duration in infos.values())
        total_runtime = time.perf_counter() - start + 1e-7

        print(f'{fn.__module__}:{fn.__qualname__} -'
              f' calls: {num_calls},'
              f' total: {Si(total_time)}s,'
              f' per-call: {Si(total_time / num_calls)}s'
              f' ({100 * total_time / total_runtime:.2f}% of module),'
              f' threads: {len(infos)}')

    def callback(duration: float):
        info = infos.setdefault(threading.get_ident(), [0, 0.])
        info[0] += 1
        info[1] += (duration - info[1]) / info[0]

    @timer(callback=callback)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    finalizer = weakref.finalize(fn, finalize, time.perf_counter())
    wrapper.finalize = finalizer  # type: ignore
    return cast(_F, functools.update_wrapper(wrapper, fn))
