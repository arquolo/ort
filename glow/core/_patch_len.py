"""Make builtin iterators compatible with `len`"""
from __future__ import annotations  # until 3.10

__all__ = ['apply']

import builtins
import functools
import operator
from collections.abc import Iterable
from itertools import (accumulate, count, cycle, islice, product, repeat,
                       starmap, tee, zip_longest)

# --------------------------------- builtins ---------------------------------

len_hint = functools.singledispatch(builtins.len)

_tee: type = tee(())[0].__class__
_iterables: list[Iterable] = [
    '', b'',
    range(0), (), [], {}, {}.keys(), {}.values(), {}.items(),
    reversed(()),
    reversed([]),
    set(),
    frozenset()
]
for _seq in _iterables:
    len_hint.register(_seq.__iter__().__class__, operator.length_hint)


@len_hint.register(zip)
def _len_zip(x):
    _, seqs = x.__reduce__()
    return min(map(len, seqs))


@len_hint.register(map)
def _len_map(x):
    _, (_fn, *seqs) = x.__reduce__()
    return min(map(len, seqs))


# --------------------------- itertools.infinite ---------------------------


@len_hint.register(count)
def _len_count(_):
    return float('+Inf')


@len_hint.register(cycle)
def _len_cycle(x):
    _, [iterable], (buf, pos) = x.__reduce__()
    if buf or len(iterable):
        return float('+Inf')
    return 0


@len_hint.register(repeat)
def _len_repeat(x):
    _, (obj, *left) = x.__reduce__()
    return left[0] if left else float('+Inf')


# ---------------------------- itertools.finite ----------------------------


@len_hint.register(accumulate)
def _len_accumulate(x):
    _, (seq, _fn), _total = x.__reduce__()
    return len(seq)


# @len_hint.register(chain)


@len_hint.register(islice)
def _len_islice(x):
    _, (src, start, *stop_step), done = x.__reduce__()
    if not stop_step:
        return 0
    stop, step = stop_step
    total = len(src) + done
    stop = total if stop is None else min(total, stop)
    return len(range(start, stop, step))


@len_hint.register(starmap)
def _len_starmap(x):
    _, (_fn, seq) = x.__reduce__()
    return len(seq)


@len_hint.register(_tee)
def _len_tee(x):
    _, [empty_tuple], (dataobject, pos) = x.__reduce__()
    _, (src, buf, none) = dataobject.__reduce__()
    return len(src) + len(buf) - pos


@len_hint.register(zip_longest)
def _len_zip_longest(x):
    _, seqs, _pad = x.__reduce__()
    return max(map(len, seqs))


# -------------------------- itertools.combinatoric -------------------------


@len_hint.register(product)
def _len_product(x):
    _, seqs, *pos = x.__reduce__()
    lens = *map(len, seqs),
    total = functools.reduce(operator.mul, lens, 1)
    if not pos:
        return total

    strides = accumulate((1, ) + lens[:0:-1], operator.mul)
    offset = sum(map(operator.mul, strides, reversed(pos[0])))
    return total - offset - 1


# @len_hint.register(permutations)
# @len_hint.register(combinations)
# @len_hint.register(combinations_with_replacement)


def apply():
    builtins.len = len_hint
