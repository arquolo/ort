from __future__ import annotations  # until 3.10

__all__ = ['coroutine', 'lock_seed', 'summary', 'trace', 'trace_module']

import functools
import gc
import inspect
import os
import random
import threading
import types
from collections import Counter
from collections.abc import Generator, Hashable
from contextlib import suppress
from typing import Callable, TypeVar, cast

import numpy as np
import wrapt

from ._import_hook import register_post_import_hook

_F = TypeVar('_F', bound=Callable[..., Generator])


def _break_on_globe(frame_infos):
    for info in frame_infos:
        yield info
        if info.function == '<module>':
            return


def whereami(skip: int = 2) -> str:
    frame_infos = _break_on_globe(inspect.stack()[skip:])
    return ' -> '.join(':'.join((
        getattr(inspect.getmodule(frame), '__name__', '[root]'),
        next((f.__qualname__ for f in gc.get_referrers(frame.f_code)
              if inspect.isfunction(f)), function),
        str(lineno),
    )) for frame, _, lineno, function, *_ in [*frame_infos][::-1])


@wrapt.decorator
def trace(fn, _, args, kwargs):
    print(
        f'<({whereami(3)})> : {fn.__module__ or ""}.{fn.__qualname__}',
        flush=True)
    return fn(*args, **kwargs)


def _set_trace(obj, seen=None, prefix=None, module=None):
    # TODO: rewrite using unittest.mock
    if isinstance(obj, types.ModuleType):
        if seen is None:
            seen = set()
            prefix = obj.__name__
        if not obj.__name__.startswith(prefix) or obj.__name__ in seen:
            return
        seen.add(obj.__name__)
        for name in dir(obj):
            _set_trace(
                getattr(obj, name), module=obj, seen=seen, prefix=prefix)

    if not callable(obj):
        return

    if not hasattr(obj, '__dict__'):
        setattr(module, obj.__qualname__, trace(obj))
        print(f'wraps "{module.__name__}:{obj.__qualname__}"')
        return

    for name in obj.__dict__:
        with suppress(AttributeError, TypeError):
            member = getattr(obj, name)
            if not callable(member):
                continue
            decorated = trace(member)

            for m in (decorated, member, obj):
                with suppress(AttributeError):
                    decorated.__module__ = m.__module__
                    break
            else:
                decorated.__module__ = getattr(module, '__name__', '')
            setattr(obj, name, decorated)
            print(f'wraps "{module.__name__}:{obj.__qualname__}.{name}"')


def trace_module(name):
    """Enables call logging for each callable inside module name"""
    register_post_import_hook(_set_trace, name)


# ---------------------------------------------------------------------------


@wrapt.decorator
def threadsafe_coroutine(fn, _, args, kwargs):
    coro = fn(*args, **kwargs)
    coro.send(None)

    class Synchronized(wrapt.ObjectProxy):
        lock = threading.RLock()

        def send(self, item):
            with self.lock:
                return self.__wrapped__.send(item)

        def __next__(self):
            return self.send(None)

    return Synchronized(coro)


@threadsafe_coroutine
def summary() -> Generator[None, Hashable, None]:
    state: Counter[Hashable] = Counter()
    while True:
        key = yield
        if key is None:
            state.clear()
            continue
        state[key] += 1
        print(dict(state), flush=True, end='\r')


def coroutine(fn: _F) -> _F:
    def wrapper(*args, **kwargs):
        coro = fn(*args, **kwargs)
        coro.send(None)
        return coro

    return cast(_F, functools.update_wrapper(wrapper, fn))


# ---------------------------------------------------------------------------


def lock_seed(seed: int) -> None:
    """Set seed for all modules: random/numpy/torch"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    def _torch_seed(torch):
        import torch
        import torch.backends.cudnn

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    register_post_import_hook(_torch_seed, 'torch')
