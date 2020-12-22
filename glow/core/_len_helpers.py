__all__ = ['as_sized']

import functools
from dataclasses import dataclass
from typing import (Callable, Generic, Iterable, Iterator, Optional, Protocol,
                    Sized, TypeVar, overload, runtime_checkable)

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_GeneratorFunc = TypeVar('_GeneratorFunc', bound=Callable[..., Iterable])
_SizeHint = Callable[..., int]


@runtime_checkable
class SizedIterable(Sized, Iterable[_T_co], Protocol[_T_co]):
    ...


@runtime_checkable
class SizedIterator(Sized, Iterator[_T_co], Protocol[_T_co]):
    ...


# ---------------------------------------------------------------------------


@dataclass(repr=False)
class _SizedIterable(Generic[_T]):
    it: Iterable[_T]
    size: int

    def __iter__(self) -> Iterator[_T]:
        return iter(self.it)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        cls = type(self.it)
        line = f'{cls.__module__}.{cls.__qualname__} object '
        if name := getattr(self.it, '__qualname__', None):
            line += f'{name} '
        line += f'at 0x{id(self.it):X} with {self.size} items'
        return f'<{line}>'


@dataclass(repr=False)
class _SizedIterator(_SizedIterable[_T]):
    it: Iterator[_T]
    size: int

    def __iter__(self) -> Iterator[_T]:
        return self

    def __next__(self) -> _T:
        self.size = max(0, self.size - 1)
        return next(self.it)


# ---------------------------------------------------------------------------


@overload
def as_sized(hint: _SizeHint) -> Callable[[_GeneratorFunc], _GeneratorFunc]:
    ...


@overload
def as_sized(gen_fn: _GeneratorFunc, hint: _SizeHint) -> _GeneratorFunc:
    ...


def as_sized(gen_fn=None, *, hint):
    """Packs generator function with size hint, thus making it sized.

    Parameters:
    - hint - callable which returns size of result iterable.
      It should have same signature as gen_fn.

    No packing occurs if hint fails with TypeError.

    >>> @as_sized(hint=len)
    ... def make_squares(it):
    ...    return (x**2 for x in it)
    ...
    >>> squares = make_squares(range(5))
    >>> len(squares)
    5
    >>> [*squares]
    [0, 1, 4, 9, 16]
    >>> len(squares)
    0
    >>> squares = make_squares(x for x in range(5))
    >>> len(squares)
    Traceback (most recent call last):
        ...
    TypeError: object of type 'generator' has no len()
    >>> [*squares]
    [0, 1, 4, 9, 16]
    """
    if gen_fn is None:
        return functools.partial(as_sized, hint=hint)

    def wrapper(*args, **kwargs):
        gen = gen_fn(*args, **kwargs)
        try:
            size = hint(*args, **kwargs)
            return (_SizedIterator if isinstance(gen, Iterator) else
                    _SizedIterable)(gen, size)
        except TypeError:
            return gen

    return functools.update_wrapper(wrapper, gen_fn)


# ---------------------------------------------------------------------------


class _PartialIter(Iterable[_T]):
    def __init__(self, hint: Optional[Callable[..., int]],
                 gen_fn: Callable[..., Iterable[_T]], *args: object,
                 **kwargs: object) -> None:
        self.gen = functools.partial(gen_fn, *args, **kwargs)
        self.hint = hint

    def __iter__(self) -> Iterator[_T]:
        return iter(self.gen())

    def __len__(self) -> int:
        if self.hint is None:
            raise TypeError('Size hint is not provided')
        return self.hint(*self.gen.args, **self.gen.keywords)

    def __repr__(self) -> str:
        line = repr(self.gen.func)
        if args := ', '.join(f'{v!r}' for v in self.gen.args):
            line += f', {args}'
        if kwargs := ','.join(
                f'{k}={v!r}' for k, v in self.gen.keywords.items()):
            line += f', {kwargs}'
        return f'<{type(self).__qualname__}({line})>'


@overload
def partial_iter(
        hint: _SizeHint = ...) -> Callable[[_GeneratorFunc], _GeneratorFunc]:
    ...


@overload
def partial_iter(gen_fn: _GeneratorFunc,
                 hint: _SizeHint = ...) -> _GeneratorFunc:
    ...


def partial_iter(gen_fn=None, *, hint=None):
    """Helper for generator functions. Adds re-iterability.

    Simplifies such code:
    ```python
    class A:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        def __iter__(self):
            <iter block using args, kwargs>
    ```
    To this:
    ```python
    @partial_iter()
    def func(*args, **kwargs):
        <iter block using args, kwargs>
    ```
    """
    if gen_fn is None:
        return functools.partial(partial_iter, hint=hint)
    return functools.update_wrapper(
        functools.partial(_PartialIter, hint, gen_fn), gen_fn)
