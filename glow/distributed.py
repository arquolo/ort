__all__ = [
    'auto_ddp', 'auto_model', 'barrier', 'get_rank', 'get_world_size',
    'reduce_if_needed'
]

import pickle
from pathlib import Path
from functools import partial, update_wrapper
from typing import Any, Callable, Optional, Protocol, Tuple, TypeVar, cast

import torch
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

# -------------------------------- primitives --------------------------------


def get_rank() -> int:
    """
    In distributed context returns the rank of current process group. otherwise
    returns -1.
    """
    return dist.get_rank() if dist.is_initialized() else -1


def get_world_size() -> int:
    """
    In distributed context returns number of processes in the current process
    group, otherwise returns 0.
    """
    return dist.get_world_size() if dist.is_initialized() else 0


def barrier(rank: Optional[int] = None) -> None:
    """Synchronize all processes"""
    if get_world_size() > 1 and (rank is None or rank == get_rank()):
        dist.barrier()


def reduce_if_needed(*tensors: torch.Tensor,
                     mean: bool = False) -> Tuple[torch.Tensor, ...]:
    """Reduce tensors across all machines"""
    if (world := get_world_size()) > 1:
        tensors = *(t.clone() for t in tensors),
        for op in [dist.all_reduce(t, async_op=True) for t in tensors]:
            op.wait()
        if mean:
            tensors = *(t / world for t in tensors),
    return tensors


# --------------------------------- wrappers ---------------------------------


def auto_model(net: nn.Module, sync_bn: bool = True) -> nn.Module:
    if (rank := get_rank()) >= 0:
        torch.cuda.set_device(rank)

        net.to(rank)
        if sync_bn:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        return nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    net.cuda()
    return (nn.parallel.DataParallel(net)
            if torch.cuda.device_count() > 1 else net)


class _TrainFn(Protocol):
    def __call__(self, __net: nn.Module, *args, **kwargs) -> Any:
        ...


_F = TypeVar('_F', bound=Callable)
_TrainFnType = TypeVar('_TrainFnType', bound=_TrainFn)


class _AutoDdp:
    def __init__(self, train_fn: _TrainFn, net: nn.Module, *args, **kwargs):
        self.train_fn = train_fn
        self.net = net
        self.args = args
        self.kwargs = kwargs
        self.ngpus = torch.cuda.device_count()

        if self.ngpus == 1:
            self._worker(None)
            return

        # ! Not tested
        # * Actually, here we can use loky.ProcessPoolExecutor, like this:
        # from . import mapped
        # jobs = mapped(
        #     self._worker, range(self.ngpus), num_workers=self.ngpus, mp=True)
        # list(jobs)
        # * Left as safe measure
        mp.spawn(self._worker, nprocs=self.ngpus)

    def _worker(self, rank: Optional[int]) -> None:
        if rank is None:
            return self.train_fn(self.net, *self.args, **self.kwargs)

        dist.init_process_group(
            backend='nccl', rank=rank, world_size=self.ngpus)
        try:
            self.train_fn(auto_model(self.net), *self.args, **self.kwargs)
        finally:
            dist.destroy_process_group()


def auto_ddp(train_fn: _TrainFnType) -> _TrainFnType:
    return cast(_TrainFnType,
                update_wrapper(partial(_AutoDdp, train_fn), train_fn))


def once_per_world(fn: _F) -> _F:
    """Call function only in rank=0 process, and share result for others"""

    def wrapper(*args, **kwargs):
        rank = get_rank()
        world = get_world_size()

        # Generate random fname and share it among whole world
        idx = torch.empty((), dtype=torch.int64).random_()
        if rank == 0:
            dist.broadcast(idx, 0)
        tmp = Path(f'/tmp/_ddp_share_{idx.item():x}.pkl')
        result = None

        if rank <= 0:  # master
            result = fn(*args, **kwargs)
            if world <= 1:
                return result  # only master exists
            with tmp.open('wb') as fp:
                pickle.dump(result, fp)

        barrier()

        if rank > 0:  # slave
            with tmp.open('rb') as fp:
                result = pickle.load(fp)

        barrier()

        if rank == 0 and world > 1:  # parent
            tmp.unlink()

        return result

    return cast(_F, update_wrapper(wrapper, fn))
