"""Patches builtin `print` function to be compatible with `tqdm`"""
__all__ = ()

import builtins
import functools
from threading import RLock
from typing import Any

import wrapt

_print = builtins.print
_lock = RLock()


@functools.wraps(_print)
def locked_print(*args, **kwargs):
    with _lock:
        _print(*args, **kwargs)


@wrapt.when_imported('tqdm')
def patch_print(tqdm: Any) -> None:
    def new_print(*args, sep=' ', end='\n', file=None, **kwargs) -> None:
        tqdm.tqdm.write(sep.join(map(str, args)), end=end, file=file)

    builtins.print = functools.update_wrapper(new_print, _print)


builtins.print = locked_print