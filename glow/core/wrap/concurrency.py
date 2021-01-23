from __future__ import annotations  # until 3.10

__all__ = [
    'stream_batched', 'call_once', 'threadlocal', 'interpreter_lock',
    'shared_call'
]

import functools
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from queue import Empty, SimpleQueue
from threading import Thread
from typing import Any, Callable, TypeVar, cast
from weakref import WeakValueDictionary

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_ZeroArgsF = TypeVar('_ZeroArgsF', bound=Callable[[], Any])


def threadlocal(fn: Callable[..., _T], *args: object,
                **kwargs: object) -> Callable[[], _T]:
    """Thread-local singleton factory, mimics `functools.partial`"""
    local_ = threading.local()

    def wrapper() -> _T:
        try:
            return local_.obj
        except AttributeError:
            local_.obj = fn(*args, **kwargs)
            return local_.obj

    return wrapper


@contextmanager
def interpreter_lock(timeout=1_000):
    """
    Prevents thread switching in underlying scope, thus makes it completely
    thread-safe. Although adds high performance penalty.

    See tests for examples.
    """
    with ExitStack() as stack:
        stack.callback(sys.setswitchinterval, sys.getswitchinterval())
        sys.setswitchinterval(timeout)
        yield


class _DeferredStack(ExitStack):
    """
    ExitStack that allows deferring.
    When return value of callback function should be accessible, use this.
    """
    def defer(self, fn: Callable[..., _T], *args, **kwargs) -> Future[_T]:
        future: Future[_T] = Future()

        def apply(future: Future[_T]) -> None:
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        self.callback(apply, future)
        return future


def call_once(fn: _ZeroArgsF) -> _ZeroArgsF:
    """Makes `fn()` callable a singleton"""
    lock = threading.RLock()

    def wrapper():
        with _DeferredStack() as stack:
            with lock:
                if fn.__future__ is None:
                    # This way setting future is protected, but fn() is not
                    fn.__future__ = stack.defer(fn)

        return fn.__future__.result()

    fn.__future__ = None  # type: ignore
    return cast(_ZeroArgsF, functools.update_wrapper(wrapper, fn))


def shared_call(fn: _F) -> _F:
    """Merges concurrent calls to `fn` with the same `args` to single one"""
    lock = threading.RLock()
    futures: WeakValueDictionary[str, Future] = WeakValueDictionary()

    def wrapper(*args, **kwargs):
        key = f'{fn}{args}{kwargs}'

        with _DeferredStack() as stack:
            with lock:
                try:
                    future = futures[key]
                except KeyError:
                    futures[key] = future = stack.defer(fn, *args, **kwargs)

        return future.result()

    return cast(_F, functools.update_wrapper(wrapper, fn))


def _batch_apply(func: Callable, args: Sequence, futures: Sequence[Future]):
    try:
        results = func(args)
        assert len(args) == len(results)
    except BaseException as exc:
        for fut in futures:
            fut.set_exception(exc)
    else:
        for fut, res in zip(futures, results):
            fut.set_result(res)


def stream_batched(func=None, *, batch_size, latency=0.1, timeout=20.):
    """
    Delays start of computation up to `latency` seconds
    in order to fill batch to batch_size items and
    send it at once to target function.
    `timeout` specifies timeout to wait results from worker.

    Simplified version of https://github.com/ShannonAI/service-streamer
    """
    if func is None:
        return functools.partial(
            stream_batched,
            batch_size=batch_size,
            latency=latency,
            timeout=timeout)

    assert callable(func)
    buf = SimpleQueue()

    def _fetch_batch():
        end_time = time.monotonic() + latency
        for _ in range(batch_size):
            try:
                yield buf.get(timeout=end_time - time.monotonic())
            except (Empty, ValueError):  # ValueError on negative timeout
                return

    def _serve_forever():
        while True:
            if batch := [*_fetch_batch()]:
                _batch_apply(func, *zip(*batch))
            else:
                time.sleep(0.001)

    def wrapper(batch):
        futures = [Future() for _ in batch]
        for item, fut in zip(batch, futures):
            buf.put((item, fut))
        return [fut.result(timeout=timeout) for fut in futures]

    Thread(target=_serve_forever, daemon=True).start()
    return functools.update_wrapper(wrapper, func)
