__all__ = ['time_this', 'timer']

import contextlib
import functools
import threading
import time
import weakref
from typing import Callable, Dict, TypeVar, cast

from ._repr import Si

_F = TypeVar('_F', bound=Callable)


@contextlib.contextmanager
def timer(name: str = 'Task', callback: Callable[[float], object] = None):
    init = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - init
        if callback is not None:
            callback(duration)
        else:
            print(f'{name} done in {duration:.4g} seconds')


def time_this(fn: _F) -> _F:
    """Log function timings at program exit"""
    infos: Dict[int, list] = {}

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
