from __future__ import annotations  # until 3.10

__all__ = ['Si', 'countable', 'mangle', 'repr_as_obj']

from collections import Callable, Counter

import wrapt


def mangle() -> Callable[[str], str | None]:
    """Appends number to already seen strings, making them distinct

    >>> mangled = mangle()
    >>> mangled('a')
    'a'
    >>> mangled('b')
    'b'
    >>> mangled('a')
    'a:1'
    """
    store: Counter[str] = Counter()

    def call(name: str) -> str | None:
        if name is None:
            return None

        seen = store[name]
        store[name] += 1
        if not seen:
            return name
        return f'{name}:{seen}'

    return call


def countable() -> Callable[[object], int]:
    """Accumulates and enumerates objects. Readable alternative to id().

    >>> id_ = countable()
    >>> id_('a')
    0
    >>> id_('b')
    1
    >>> id_('a')
    0
    """
    instances: dict[int, int] = {}
    return lambda obj: instances.setdefault(id(obj), len(instances))


def repr_as_obj(d: dict) -> str:
    """Returns pretty representation of dict.

    >>> repr_as_obj({'a': 1, 'b': 2})
    'a=1, b=2'
    """
    return ', '.join(f'{key}={value!r}' for key, value in d.items())


class Si(wrapt.ObjectProxy):
    """Wrapper for numbers with human-readable formatting.

    Use metric prefixes:
    >>> s = Si(10 ** 6)
    >>> s
    Si(1M)
    >>> print(s)
    1M

    Use binary prefixes:
    >>> print(Si.bits(2 ** 20))
    1MiB

    .. _Human readable bytes count
       https://programming.guide/java/formatting-byte-size-to-human-readable-format.html
    """
    _prefixes = 'qryzafpnum kMGTPEZYRQ'
    _prefixes_bin = _prefixes[_prefixes.index(' '):].upper()

    def __init__(self, value: float | int = 0, _si: bool = True):
        super().__init__(value)
        self._self_si = _si

    @classmethod
    def bits(cls, value: float | int = 0) -> Si:
        return cls(value, _si=False)

    def __str__(self):
        x = self.__wrapped__
        if x == 0:
            return '0'

        unit, prefixes = ((1000, self._prefixes) if self._self_si else
                          (1024, self._prefixes_bin))
        unit_thres = unit - 0.5
        origin = prefixes.find(' ') + 1

        x *= unit ** origin
        for prefix in prefixes:  # noqa: B008
            x /= unit
            if -unit_thres < x < unit_thres:
                break
        else:
            prefix = prefixes[-1]

        precision = '.0f' if x >= 99.95 else '.3g'
        if not self._self_si:
            prefix = f'{prefix}iB' if prefix.strip() else 'B'
        return f'{x:{precision}}{prefix.strip()}'

    def __repr__(self):
        return f'{type(self).__name__}({self})'
