__all__ = [
    'Metric', 'Lambda', 'Scores', 'Staged', 'compose', 'to_index', 'to_prob'
]

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Callable, Dict, Generator, Protocol, Sequence, Tuple,
                    Union, overload)

import torch

from .. import coroutine

TensorDict = Dict[str, torch.Tensor]


@dataclass
class Scores:
    scalars: Dict[str, Union[float, int]] = field(default_factory=dict)
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, mapping: TensorDict) -> 'Scores':
        obj = cls()
        for k, v in mapping.items():
            if v.numel() == 1:
                obj.scalars[k] = v.item()
            else:
                obj.tensors[k] = v
        return obj


class _MetricFn(Protocol):
    def __call__(self, pred, true) -> torch.Tensor:
        ...


class Metric(ABC):
    """Base class for metric"""
    @abstractmethod
    def __call__(self, pred, true) -> torch.Tensor:
        raise NotImplementedError

    def collect(self, state) -> Dict[str, torch.Tensor]:
        raise state


class Lambda(Metric):
    """Wraps arbitary loss function to metric"""
    fn: _MetricFn

    @overload
    def __init__(self, fn: Callable, name: str):
        ...

    @overload
    def __init__(self, fn: _MetricFn, name: None = ...):
        ...

    def __init__(self, fn, name=None):
        self.fn = fn
        self.name = fn.__name__ if name is None else name  # type: ignore

    def __call__(self, pred, true) -> torch.Tensor:
        return self.fn(pred, true)

    def collect(self, state):
        return {self.name: state}


class Staged(Metric):
    """Makes metric a "producer": applies multiple functions to its "state" """
    def __init__(self, **funcs: Callable[[torch.Tensor], torch.Tensor]):
        self.funcs = funcs

    def collect(self, state):
        return {key: fn(state) for key, fn in self.funcs.items()}


def to_index(pred, true) -> Tuple[int, torch.LongTensor, torch.LongTensor]:
    """
    Convert `pred` of logits with shape [B, C, ...] to [B, ...] of indices.
    Drop bad indices.
    """
    c = pred.shape[1]
    pred = pred.argmax(dim=1)

    if true.min() < 0 or true.max() >= c:
        mask = (true >= 0) & (true < c)
        true = true[mask][None]
        pred = pred[mask][None]

    return c, pred, true


def to_prob(pred, true) -> Tuple[int, torch.Tensor, torch.LongTensor]:
    """
    Convert `pred` of logits with shape [B, C, ...] to probs.
    Drop bad indices.
    """
    b, c = pred.shape[:2]
    pred = pred.softmax(dim=1)

    if true.min() < 0 or true.max() >= c:
        true = true.view(-1)
        pred = pred.view(b, c, -1).permute(0, 2, 1).view(-1, c)

        mask = (true >= 0) & (true < c)
        true = true[mask]
        pred = pred[mask]

    return c, pred, true


@coroutine
def _batch_averaged(
        fn: Metric) -> Generator[TensorDict, Sequence[torch.Tensor], None]:
    assert isinstance(fn, Metric)
    args = yield {}
    state = torch.as_tensor(fn(*args))
    for step in itertools.count(2):
        args = yield fn.collect(state)
        state += (torch.as_tensor(fn(*args)) - state) / step


@coroutine
def compose(*fns: Metric) -> Generator[Scores, Sequence[torch.Tensor], None]:
    updates = *map(_batch_averaged, fns),
    args = yield Scores()
    while True:
        scores = {k: v for u in updates for k, v in u.send(args).items()}
        args = yield Scores.from_dict(scores)
