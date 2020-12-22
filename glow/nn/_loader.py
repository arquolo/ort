__all__ = ['make_loader']

import os
import warnings
from collections import abc
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, Union

import torch
from torch.utils.data import Dataset, IterableDataset, Sampler
from torch.utils.data._utils import worker as torch_worker

from .. import buffered, chunked, mapped, roundrobin
from ..core._parallel import _get_executor
from ..distributed import get_rank, get_world_size

_NUM_CPUS: int = os.cpu_count() or 1


class _CollateFn(Protocol):
    def __call__(self, __items: abc.Sequence) -> Any:
        ...


def default_collate(batch: Sequence[tuple]) -> Any:
    return tuple(
        torch.stack([torch.as_tensor(item) for item in row])
        for row in zip(*batch))


def pin_memory(data):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    if hasattr(data, 'pin_memory'):
        return data._pin_memory()

    if isinstance(data, (str, bytes)):
        return data
    if isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory(sample) for sample in data))
    if isinstance(data, abc.Sequence):
        return [pin_memory(sample) for sample in data]

    if isinstance(data, abc.Mapping):
        return {k: pin_memory(sample) for k, sample in data.items()}

    return data


# ------------------------------- base loader -------------------------------


@dataclass(frozen=True)
class _BaseLoader:
    batch_size: int
    num_workers: int
    collate_fn: _CollateFn
    pin_memory: bool
    dataset: Dataset

    def _iter_samples(self) -> abc.Iterator:
        raise NotImplementedError

    def __iter__(self) -> abc.Iterator:
        batches = chunked(self._iter_samples(), self.batch_size)
        batches = map(self.collate_fn, batches)
        if not self.pin_memory:
            return batches
        return iter(mapped(pin_memory, batches, num_workers=1))

    def __len__(self) -> int:
        raise NotImplementedError


# ---------------------- loader for map-style datasets ----------------------


@dataclass(frozen=True)
class _MapLoader(_BaseLoader):
    sampler: Sampler
    mp: bool
    chunksize: Optional[int]

    def _iter_samples(self) -> abc.Iterator:
        num_workers = self.num_workers
        if (world := get_world_size()) > 1:
            num_workers //= world

        if not num_workers:
            return map(self.dataset.__getitem__, self.sampler)

        samples = mapped(
            self.dataset.__getitem__,
            self.sampler,
            mp=self.mp,
            chunksize=self.chunksize,
            num_workers=num_workers)
        return iter(samples)

    def __len__(self) -> int:
        indices = range(0, len(self.sampler), self.batch_size)  # type: ignore
        return len(indices)

    def set_epoch(self, epoch: int):
        if isinstance(self.sampler, _AutoSampler):
            self.sampler.set_epoch(epoch)


# -------------------- loader for iterable-style datasets --------------------


@dataclass(frozen=True)
class _Worker:
    dataset: IterableDataset
    id: int  # noqa: A003, shadowing builtin `id`, false positive
    num_workers: int
    seed: Optional[int] = None

    def __iter__(self) -> abc.Iterator:
        torch_worker._worker_info = self
        try:
            yield from self.dataset
        finally:
            torch_worker._worker_info = None


@dataclass(frozen=True)
class _IterableLoader(_BaseLoader):
    dataset: IterableDataset

    def _iter_samples(self) -> abc.Iterator:
        if not self.num_workers:
            yield from buffered(self.dataset)
            return

        seed = torch.empty((), dtype=torch.int64).random_().item()
        workers = [
            _Worker(self.dataset, idx, self.num_workers, int(seed))
            for idx in range(self.num_workers)
        ]
        with _get_executor(self.num_workers, True) as executor:
            yield from roundrobin(*(buffered(w, mp=executor) for w in workers))

    def __len__(self) -> int:
        indices = range(0, len(self.dataset), self.batch_size)  # type: ignore
        return len(indices)


# --------------------------------- samplers ---------------------------------


class _IAutoSampler(Sampler):
    def __init__(self, source: Union[Dataset, Sampler]) -> None:
        if not isinstance(source, abc.Sized):
            raise TypeError('Argument should have length')

        self.source = source
        self.epoch = 0
        self.seed = int(torch.empty((), dtype=torch.int64).random_().item())

    def _indices(self) -> list:
        raise NotImplementedError

    def __iter__(self) -> abc.Iterator:
        indices = self._indices()

        if (world := get_world_size()) > 1:
            if remainder := (len(indices) % world):
                indices += indices[:world - remainder]
            indices = indices[get_rank()::world]

        if len(indices) != len(self):
            raise RuntimeError(f'{len(indices)} vs {len(self)}')

        return iter(indices)

    def __len__(self) -> int:
        world = get_world_size() or 1
        return (len(self.source) + world - 1) // world

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _AutoSampler(_IAutoSampler):
    def __init__(self, dataset: Dataset, shuffle: bool = False):
        if not isinstance(dataset, abc.Sized):
            raise TypeError('Argument sampler should have length')

        super().__init__(dataset)
        self.shuffle = shuffle

    def _indices(self) -> list:
        if not self.shuffle:
            return list(range(len(self.source)))

        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.source), generator=rng).tolist()


class _AutoSamplerProxy(_IAutoSampler):
    def _indices(self) -> list:
        torch.manual_seed(self.seed + self.epoch)
        return [*self.source]  # type: ignore


# ----------------------------- factory function -----------------------------


def make_loader(dataset: Dataset,
                batch_size: int,
                shuffle: bool = False,
                sampler: Sampler = None,
                num_workers: int = _NUM_CPUS,
                collate_fn: _CollateFn = default_collate,
                pin_memory: bool = False,
                multiprocessing: bool = True,
                chunk_from_batch: bool = False) -> _BaseLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable
    over the given dataset.

    The data loader supports both map-style and iterable-style datasets with
    single- or multi-process loading, customizing loading order and
    automatic batching (collation) and memory pinning.

    Yields batches of batch_size from dataset in order from sampler.

    Differences from torch.utils.data.DataLoader:
    - Support of threadpool backend for map-style datasets.
    - Automatically adjusts chunksize of data for passing to the workers to
      reduce IPC overhead when multiple processes are used.
    - Automatically adjusts number of workers in distributed context for
      map-style datasets.
    - Use set_epoch() method before __iter__() for reprocucibility.

    Parameters:
    - batch_size - size of batch, each workers computes batch independently.
    - workers - Count of workers, by default all hardware threads are occupied.
    - multiprocessing - whether to use processes or threads.
    - chunk_from_batch - Set count of samples to pass each worker. If set
      then chunksize will be equal to batch_size, otherwise it will be
      estimated automatically.
    """
    if isinstance(dataset, IterableDataset):
        if shuffle or sampler is not None:
            raise ValueError(
                'Loader with IterableDataset: sampler/shuffle options are '
                'not supported')

        if not multiprocessing and num_workers != 0:
            warnings.warn(
                'Loader with IterableDataset: ThreadPool is not supported. '
                'Setting num_workers to 0')
            num_workers = 0

        if get_world_size() > 1:
            raise ValueError(
                'Loader with IterableDataset: distributed context is not '
                'supported')

        return _IterableLoader(batch_size, num_workers, collate_fn, pin_memory,
                               dataset)

    else:
        if shuffle and sampler is not None:
            raise ValueError(
                'Loader with MapDataset: sampler option is mutually exclusive '
                'with shuffle')

        if sampler is None:
            sampler = _AutoSampler(dataset, shuffle=shuffle)
        else:
            sampler = _AutoSamplerProxy(sampler)

        chunksize = batch_size if multiprocessing and chunk_from_batch else None
        return _MapLoader(batch_size, num_workers, collate_fn, pin_memory,
                          dataset, sampler, multiprocessing, chunksize)
