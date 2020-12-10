__all__ = ['make_loader']

import os
from typing import Any, Callable, Iterator, NamedTuple, Optional, Tuple, Union

import torch
from torch.utils.data import (Dataset, IterableDataset, RandomSampler, Sampler,
                              SequentialSampler)
from torch.utils.data._utils import worker as torch_worker

from .. import buffered, chunked, mapped, partial_iter, roundrobin
from ..core._parallel import _get_executor

_CollateFn = Callable[[Any], Tuple[torch.Tensor, ...]]
_NUM_CPUS: int = os.cpu_count() or 1


def _default_collate(batch) -> Tuple[torch.Tensor, ...]:
    return tuple(
        torch.stack([torch.as_tensor(item) for item in row])
        for row in zip(*batch))


# ---------------------- loader for map-style datasets ----------------------


class _MapLoader(NamedTuple):
    dataset: Dataset
    batch_size: int
    sampler: Sampler
    num_workers: int
    mp: bool
    collate_fn: Optional[_CollateFn]
    chunksize: Optional[int]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        samples = mapped(
            self.dataset.__getitem__,
            self.sampler,
            mp=self.mp,
            chunksize=self.chunksize,
            num_workers=self.num_workers)
        batches = chunked(samples, self.batch_size)
        return (batches if self.collate_fn is None  # type: ignore
                else map(self.collate_fn, batches))

    def __len__(self) -> int:
        indices = range(0, len(self.sampler), self.batch_size)  # type: ignore
        return len(indices)


# -------------------- loader for iterable-style datasets --------------------


class _WorkerInfo(NamedTuple):
    dataset: IterableDataset
    id: int  # noqa: A003, shadowing builtin `id`, false positive
    num_workers: int
    seed: Optional[int] = None


def _worker_fn(dataset: IterableDataset,
               idx: int,
               num_workers: int,
               seed: int = None):
    torch_worker._worker_info = _WorkerInfo(dataset, idx, num_workers, seed)
    return iter(dataset)


class _IterableLoader(NamedTuple):
    dataset: IterableDataset
    batch_size: int
    num_workers: int
    mp: bool
    collate_fn: Optional[_CollateFn]
    seed: Optional[int]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        with _get_executor(self.num_workers, self.mp) as executor:
            workers = [
                buffered(
                    partial_iter(_worker_fn)(self.dataset, idx,
                                             self.num_workers, self.seed),
                    mp=executor) for idx in range(self.num_workers)
            ]
            batches = chunked(roundrobin(*workers), self.batch_size)
            if self.collate_fn is None:
                yield from batches  # type: ignore
            else:
                yield from map(self.collate_fn, batches)

    def __len__(self) -> int:
        indices = range(0, len(self.dataset), self.batch_size)  # type: ignore
        return len(indices)


# ----------------------------- factory function -----------------------------


def make_loader(
    dataset: Dataset,
    batch_size: int,
    sampler: Sampler = None,
    num_workers: int = _NUM_CPUS,
    multiprocessing: bool = True,
    shuffle: bool = False,
    collate_fn: Optional[_CollateFn] = _default_collate,
    chunk_from_batch: bool = False,
    seed: int = None,
) -> Union[_MapLoader, _IterableLoader]:
    """Yields batches of batch_size from dataset in order from sampler.

    Parameters:
    - batch_size - size of batch, each workers computes batch independently.
    - workers - Count of workers, by default all hardware threads are occupied.
    - multiprocessing - whether to use processes or threads.
    - chunk_from_batch - Set count of samples to pass each worker. If set
      then chunksize will be equal to batch_size, otherwise it will be
      estimated automatically.
    """
    if isinstance(dataset, IterableDataset):
        if sampler is not None:
            raise ValueError('For IterableDataset sampler is not supported')
        return _IterableLoader(dataset, batch_size, num_workers,
                               multiprocessing, collate_fn, seed)

    if sampler is None:
        sampler_fn = RandomSampler if shuffle else SequentialSampler
        sampler = sampler_fn(dataset)  # type: ignore

    chunksize = batch_size if multiprocessing and chunk_from_batch else None
    return _MapLoader(dataset, batch_size, sampler, num_workers,
                      multiprocessing, collate_fn, chunksize)
