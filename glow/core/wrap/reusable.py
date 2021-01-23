from __future__ import annotations  # until 3.10

__all__ = ['Reusable']

import asyncio
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Generic, Protocol, TypeVar

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)


def make_loop() -> asyncio.AbstractEventLoop:
    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    threading.Thread(target=start_loop, args=(loop, ), daemon=True).start()
    return loop


def _call_in_loop(fn: Callable[..., _T],
                  loop: asyncio.AbstractEventLoop) -> _T:
    async def callee():
        return fn()

    return asyncio.run_coroutine_threadsafe(callee(), loop=loop).result()


class _Factory(Protocol[_T_co]):
    def __call__(self) -> _T_co:
        ...


@dataclass
class Reusable(Generic[_T]):
    _loop: ClassVar[asyncio.AbstractEventLoop] = make_loop()
    _lock: asyncio.Lock = field(init=False)
    factory: _Factory[_T]
    delay: float
    finalize: Callable[[_T], None] | None = None
    _deleter: asyncio.TimerHandle | None = None
    _box: list[_T] = field(default_factory=list)

    def __post_init__(self):
        self._lock = _call_in_loop(asyncio.Lock, self._loop)

    def get(self) -> _T:
        """Returns inner object, or recreates it"""
        fut = asyncio.run_coroutine_threadsafe(self._get(), loop=self._loop)
        return fut.result()

    def _finalize(self, ref):
        obj: _T | None = ref()
        if obj is not None and self.finalize is not None:
            self.finalize(obj)

    async def _get(self) -> _T:
        async with self._lock:
            # reschedule finalizer
            if self._deleter is not None:
                self._deleter.cancel()
            self._deleter = self._loop.call_later(self.delay, self._box.clear)

            # recreate object, if nessessary
            if not self._box:
                obj = self.factory()
                weakref.finalize(obj, self._finalize, weakref.ref(obj))
                self._box.append(obj)

            return self._box[0]
