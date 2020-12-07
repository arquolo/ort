__all__ = [
    'call_once', 'memoize', 'shared_call', 'stream_batched', 'threadlocal'
]

import enum
import functools
import sys
import threading
import time
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from dataclasses import InitVar, asdict, dataclass, field
from queue import Empty, SimpleQueue
from threading import RLock, Thread
from typing import (Any, Callable, ClassVar, Dict, Generic, Hashable, KeysView,
                    List, Literal, MutableMapping, Optional, Sequence, Type,
                    TypeVar, Union, cast)
from weakref import WeakValueDictionary

from ._repr import Si
from ._sizeof import sizeof

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_ZeroArgsF = TypeVar('_ZeroArgsF', bound=Callable[[], Any])
_BatchF = TypeVar('_BatchF', bound=Callable[[Sequence], list])
_Policy = Literal['raw', 'lru', 'mru']
_KeyFn = Callable[..., Hashable]


class _Empty(enum.Enum):
    token = 0


_empty = _Empty.token

# --------------------------------- helpers ---------------------------------


def _key_fn(*args, **kwargs) -> str:
    return f'{args}{kwargs}'


@dataclass(frozen=True)
class _Job:
    item: Any
    future: Future = field(default_factory=Future)


def _dispatch(func: _BatchF, jobs: Sequence[_Job]) -> None:
    try:
        results = func([job.item for job in jobs])
        assert len(results) == len(jobs)

    except BaseException as exc:
        for job in jobs:
            job.future.set_exception(exc)

    else:
        for job, res in zip(jobs, results):
            job.future.set_result(res)


class _DeferredStack(ExitStack):
    """
    ExitStack that allows deferring.
    When return value of callback function should be accessible, use this.
    """
    def defer(self, fn: Callable[..., _T], *args, **kwargs) -> 'Future[_T]':
        future: 'Future[_T]' = Future()

        def apply(future: 'Future[_T]') -> None:
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        self.callback(apply, future)
        return future


@contextmanager
def _interpreter_lock(timeout=1_000):
    """
    Prevents thread switching in underlying scope, thus makes it completely
    thread-safe. Although adds high performance penalty.

    See tests for examples.
    """
    with ExitStack() as stack:
        stack.callback(sys.setswitchinterval, sys.getswitchinterval())
        sys.setswitchinterval(timeout)
        yield


# -------------------------- zero argument wrappers --------------------------


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


# ----------------------------- caching helpers -----------------------------


@dataclass(repr=False)
class _Node(Generic[_T]):
    __slots__ = ('value', 'size')
    value: _T
    size: Si

    def __repr__(self) -> str:
        return f'({self.value} / {self.size})'


@dataclass(repr=False, eq=False)
class Stats:
    hits: int = 0
    misses: int = 0
    drops: int = 0

    def __bool__(self):
        return any(asdict(self).values())

    def __repr__(self):
        data = ', '.join(f'{k}={v}' for k, v in asdict(self).items() if v)
        return f'{type(self).__name__}({data})'


class _IStore(Generic[_T]):
    def __len__(self) -> int:
        raise NotImplementedError

    def store_clear(self) -> None:
        raise NotImplementedError

    def store_get(self, key: Hashable) -> Optional[_Node[_T]]:
        raise NotImplementedError

    def store_set(self, key: Hashable, node: _Node[_T]) -> None:
        raise NotImplementedError

    def can_swap(self, size: Si) -> bool:
        raise NotImplementedError


@dataclass(repr=False)
class _InitializedStore:
    capacity_int: InitVar[int]

    size: Si = Si.bits(0)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self, capacity_int: int):
        self.capacity = Si.bits(capacity_int)


@dataclass(repr=False)
class _DictMixin(_InitializedStore, _IStore[_T]):
    lock: RLock = field(default_factory=RLock)

    def clear(self):
        with self.lock:
            self.store_clear()
            self.size = Si.bits()

    def keys(self) -> KeysView:
        raise NotImplementedError

    def __getitem__(self, key: Hashable) -> Union[_T, _Empty]:
        with self.lock:
            if node := self.store_get(key):
                self.stats.hits += 1
                return node.value
        return _empty

    def __setitem__(self, key: Hashable, value: _T) -> None:
        with self.lock:
            self.stats.misses += 1
            size = sizeof(value)
            if (self.size + size <= self.capacity) or self.can_swap(size):
                self.store_set(key, _Node(value, size))
                self.size += size


@dataclass(repr=False)
class _ReprMixin(_InitializedStore, _IStore[_T]):
    refs: ClassVar[MutableMapping[int, '_ReprMixin']] = WeakValueDictionary()

    def __post_init__(self, capacity_int: int) -> None:
        super().__post_init__(capacity_int)
        self.refs[id(self)] = self

    def __repr__(self) -> str:
        line = (
            f'{type(self).__name__}'
            f'(items={len(self)}, used={self.size}, total={self.capacity})')
        if self.stats:
            line += f'-{self.stats}'
        return line

    @classmethod
    def status(cls) -> str:
        with _interpreter_lock():
            return '\n'.join(f'{id_:x}: {value!r}'
                             for id_, value in sorted(cls.refs.items()))


@dataclass(repr=False)
class _Store(_ReprMixin[_T], _DictMixin[_T]):
    store: Dict[Hashable, _Node[_T]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.store)

    def keys(self) -> KeysView:
        return self.store.keys()

    def store_clear(self) -> None:
        self.store.clear()

    def store_get(self, key: Hashable) -> Optional[_Node[_T]]:
        return self.store.get(key)

    def store_set(self, key: Hashable, node: _Node[_T]) -> None:
        self.store[key] = node


class _HeapCache(_Store[_T]):
    def can_swap(self, size: Si) -> bool:
        return False


class _LruCache(_Store[_T]):
    drop_recent = False

    def store_get(self, key: Hashable) -> Optional[_Node[_T]]:
        if node := self.store.pop(key, None):
            self.store[key] = node
            return node
        return None

    def can_swap(self, size: Si) -> bool:
        if size > self.capacity:
            return False

        while self.size + size > self.capacity:
            if self.drop_recent:
                self.size -= self.store.popitem()[1].size
            else:
                self.size -= self.store.pop(next(iter(self.store))).size
            self.stats.drops += 1
        return True


class _MruCache(_LruCache[_T]):
    drop_recent = True


# ------------------------------ cache wrappers ------------------------------


def _memoize(cache: _DictMixin, key_fn: _KeyFn, func: _F) -> _F:
    def wrapper(*args, **kwargs):
        key = key_fn(*args, **kwargs)

        if (value := cache[key]) is not _empty:
            return value

        cache[key] = value = func(*args, **kwargs)
        return value

    wrapper.cache = cache  # type: ignore
    return cast(_F, functools.update_wrapper(wrapper, func))


def _memoize_batched(cache: _DictMixin, key_fn: _KeyFn,
                     func: _BatchF) -> _BatchF:
    assert callable(func)
    lock = RLock()
    pending: Dict[Hashable, _Job] = {}

    def _safe_dispatch():
        with lock:
            keys, jobs = zip(*pending.items())
            pending.clear()

        _dispatch(func, jobs)

        for key, job in zip(keys, jobs):
            if not job.future.exception():
                cache[key] = job.future.result()

    def _get_future(stack: ExitStack, key: Hashable, item: Any) -> Future:
        if job := pending.get(key):
            return job.future

        future = Future()  # type: ignore

        if (value := cache[key]) is not _empty:
            future.set_result(value)
            return future

        pending[key] = _Job(item, future)
        if len(pending) == 1:
            stack.callback(_safe_dispatch)

        return future

    def wrapper(items: Sequence) -> list:
        keyed_items = [(key_fn(item), item) for item in items]

        results = {}
        with lock:
            for i, (key, item) in enumerate(keyed_items):
                if job := pending.get(key):
                    job.idx.append(i)
                elif (value := cache[key]) is not _empty:
                    results[i] = value
                else:
                    pending[key] = _Job(item)

            keys: List[Hashable]
            jobs: List[_Job]
            keys, jobs = zip(*pending.items())  # type: ignore
            pending.clear()

        if jobs:
            _dispatch(func, jobs)

        with lock:
            for key, job in zip(keys, jobs):
                if job.future.exception():
                    continue
                cache[key] = r = job.future.result()
                results.update((i, r) for i in job.idx)

        with ExitStack() as stack:
            with lock:
                futs = [_get_future(stack, *ki) for ki in keyed_items]
        return [fut.result() for fut in futs]

    wrapper.cache = cache  # type: ignore
    return cast(_BatchF, functools.update_wrapper(wrapper, func))


def memoize(
    capacity: int,
    *,
    batched: bool = False,
    policy: _Policy = 'raw',
    key_fn: _KeyFn = _key_fn
) -> Union[Callable[[_F], _F], Callable[[_BatchF], _BatchF]]:
    """Returns dict-cache decorator.

    Parameters:
    - capacity - size in bytes.
    - policy - eviction policy, either "raw" (no eviction), or "lru"
      (evict oldest), or "mru" (evict most recent).
    """
    if not capacity:
        return lambda fn: fn

    caches: Dict[str, Type[_Store]] = {
        'raw': _HeapCache,
        'lru': _LruCache,
        'mru': _MruCache,
    }
    if cache_cls := caches.get(policy):
        mem_fn = _memoize_batched if batched else _memoize
        return functools.partial(mem_fn, cache_cls(capacity), key_fn)

    raise ValueError(f'Unknown policy: "{policy}". '
                     f'Only "{set(caches)}" are available')


def shared_call(fn: _F) -> _F:
    """Merges concurrent calls to `fn` with the same `args` to single one"""
    lock = RLock()
    futures: 'WeakValueDictionary[str, Future]' = WeakValueDictionary()

    def wrapper(*args, **kwargs):
        key = _key_fn(*args, **kwargs)

        with _DeferredStack() as stack:
            with lock:
                try:
                    future = futures[key]
                except KeyError:
                    futures[key] = future = stack.defer(fn, *args, **kwargs)

        return future.result()

    return cast(_F, functools.update_wrapper(wrapper, fn))


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
    inputs = SimpleQueue()

    def _serve_forever():
        while True:
            jobs = []
            end_time = time.monotonic() + latency
            while (len(jobs) < batch_size and
                   (time_left := end_time - time.monotonic()) > 0):
                try:
                    jobs.append(inputs.get(timeout=time_left))
                except Empty:
                    if not jobs:
                        time.sleep(0.001)
                    break
            _dispatch(func, jobs)

    def wrapper(items):
        jobs = [_Job(item) for item in items]
        for job in jobs:
            inputs.put(job)
        return [job.future.result(timeout=timeout) for job in jobs]

    Thread(target=_serve_forever, daemon=True).start()
    return functools.update_wrapper(wrapper, func)
