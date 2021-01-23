from __future__ import annotations  # until 3.10

__all__ = ['Stepper']

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
import torch.cuda.amp
import torch.nn as nn
import torch.optim
from tqdm.auto import tqdm

from .. import metrics as m
from .amp import OptContext, get_amp_context


@dataclass
class Stepper:
    net: nn.Module
    opt: torch.optim.Optimizer
    criterion: Callable
    metrics: Iterable[m.Metric]
    device: torch.device
    fp16: bool = False
    _ctx: OptContext = field(init=False)  # type: ignore

    def __post_init__(self):
        self._ctx = get_amp_context(
            self.net, self.opt, fp16=self.fp16, retry_on_inf=True)

    def _step(self, data: torch.Tensor, target: torch.Tensor,
              is_train: bool) -> tuple[torch.Tensor, ...]:
        target = target
        with torch.cuda.amp.autocast(self.fp16):
            out = self.net(data.to(self.device))
        if is_train:
            with self._ctx:
                self._ctx.backward(self.criterion(out, target))
        return out.detach(), target

    def run(self, loader, pbar: tqdm, is_train: bool = True) -> m.Scores:
        meter = m.compose(*self.metrics)
        scores = m.Scores()

        was_train = self.net.training
        self.net.train(is_train)

        for data, target in loader:
            with torch.set_grad_enabled(is_train):
                out = self._step(
                    data.to(self.device, non_blocking=True),
                    target.to(self.device, non_blocking=True),
                    is_train=is_train)
            scores = meter.send(out)

            pbar.set_postfix(scores.scalars)
            pbar.update()

        self.net.train(was_train)
        return scores
