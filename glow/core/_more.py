from __future__ import annotations  # until 3.10

__all__ = [
    'as_iter', 'chunked', 'eat', 'ichunked', 'roundrobin', 'sliced', 'windowed'
]

import enum
import threading
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain, cycle, islice, repeat, tee
from typing import TypeVar

from ._len_helpers import _SizedIterator, as_sized


class _Empty(enum.Enum):
    token = 0


_T = TypeVar('_T')
_empty = _Empty.token


def as_iter(obj: Iterable[_T] | _T | None,
            times: int = None) -> Iterable[_T]:
    """Make iterator from object"""
    if obj is None:
        return ()
    if isinstance(obj, Iterable):
        return islice(obj, times)
    return repeat(obj) if times is None else repeat(obj, times)


def windowed(it: Iterable[_T], size: int) -> Iterator[Sequence[_T]]:
    """Retrieve overlapped windows from iterable.

    >>> [*windowed(range(5), 3)]
    [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    """
    return zip(*(islice(it_, start, None)
                 for start, it_ in enumerate(tee(it, size))))


def sliced(seq: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    """Split sequence to slices of at most size items each.

    >>> s = sliced(range(10), 3)
    >>> len(s)
    4
    >>> [*s]
    [range(0, 3), range(3, 6), range(6, 9), range(9, 10)]
    """
    offsets = range(len(seq) + size)
    slices = map(slice, offsets[0::size], offsets[size::size])
    return map(seq.__getitem__, slices)


def chunk_hint(it, size):
    return len(range(0, len(it), size))


@as_sized(hint=chunk_hint)
def chunked(it: Iterable[_T], size: int) -> Iterator[Sequence[_T]]:
    """Split iterable to chunks of at most size items each.
    Each next() on result will advance passed iterable to size items.

    >>> s = chunked(range(10), 3)
    >>> len(s)
    4
    >>> [*s]
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
    """
    chunks = map(islice, repeat(iter(it)), repeat(size))
    return iter(map(tuple, chunks).__next__, ())  # type: ignore


@as_sized(hint=chunk_hint)
def ichunked(it: Iterable[_T], size: int) -> Iterator[Iterator[_T]]:
    """Split iterable to chunks of at most size items each.

    Does't consume items from passed iterable to return complete chunk
    unlike chunked, as yields iterators, not sequences.

    >>> s = ichunked(range(10), 3)
    >>> len(s)
    4
    >>> [[*chunk] for chunk in s]
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    head = iter(it)
    while (item := next(head, _empty)) is not _empty:
        it1 = islice(chain([item], head), size)  # Restore iterable
        it1, it2 = tee(it1)  # Fork to keep not-yet-consumed
        yield _SizedIterator(it1, size)
        eat(it2)  # Advance to head[size:]


def eat(iterable: Iterable, daemon: bool = False) -> None:
    """Consume iterable, daemonize if needed (move to background thread)"""
    if not daemon:
        deque(iterable, 0)
    else:
        threading.Thread(target=deque, args=(iterable, 0), daemon=True).start()


@as_sized(hint=lambda *it: sum(map(len, it)))
def roundrobin(*iterables: Iterable[_T]) -> Iterator[_T]:
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    iters = cycle(map(iter, iterables))
    for pending in range(len(iterables))[::-1]:
        yield from map(next, iters)
        iters = cycle(islice(iters, pending))
