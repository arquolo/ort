from __future__ import annotations  # until 3.10

__all__ = ['sizeof']

import functools
import sys
from collections import abc
from enum import Enum
from inspect import isgetsetdescriptor, ismemberdescriptor

import numpy as np

from ._import_hook import when_imported
from ._repr import Si


def for_unseen(fn, default=Si.bits):
    """Protection from self-referencing"""
    def wrapper(obj, seen: set[int] = None) -> Si:
        if seen is None:
            seen = set()
        id_ = id(obj)
        if id_ in seen:
            return default()

        seen.add(id_)
        return fn(obj, seen)

    return functools.update_wrapper(wrapper, fn)


@functools.singledispatch
@for_unseen
def sizeof(obj, seen: set[int] = None) -> Si:
    """Computes size of object, no matter how complex it is.

    Inspired by
    [PySize](https://github.com/bosswissam/pysize/blob/master/pysize.py)
    """
    size = sys.getsizeof(obj)
    if isinstance(obj, (str, bytes, bytearray, Si, Enum)):
        return Si.bits(size)

    if hasattr(obj, '__dict__'):
        for d in (vars(cl)['__dict__']
                  for cl in obj.__class__.__mro__ if '__dict__' in vars(cl)):
            if isgetsetdescriptor(d) or ismemberdescriptor(d):
                size += sizeof(vars(obj), seen)
            break

    if isinstance(obj, dict):
        size += sum(sizeof(k, seen) for k in obj.keys())
        size += sum(sizeof(v, seen) for v in obj.values())
    elif isinstance(obj, abc.Collection):
        size += sum(sizeof(item, seen) for item in obj)

    if hasattr(obj, '__slots__'):
        size += sum(
            sizeof(getattr(obj, slot, None), seen)
            for class_ in type(obj).mro()
            for slot in getattr(class_, '__slots__', ()))
    return Si.bits(size)


@for_unseen
def _sizeof_numpy(obj, _=None) -> Si:
    return Si.bits(sys.getsizeof(obj if obj.base is None else obj.base))


@for_unseen
def _sizeof_torch(obj, _=None) -> Si:
    size = sys.getsizeof(obj)
    if not obj.is_cuda:
        size += obj.numel() * obj.element_size()
    return Si.bits(size)  # TODO: test, maybe useless when grads are attached


sizeof.register(np.ndarray, _sizeof_numpy)
when_imported('torch')(
    lambda torch: sizeof.register(torch.Tensor, _sizeof_torch))
