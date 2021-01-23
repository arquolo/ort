"""Utilities to train and eval nn.Module in half-precision mode (float16)."""

from __future__ import annotations  # until 3.10

__all__ = ['get_amp_context']

import warnings
from collections import abc
from functools import partial

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim

_MIN_SCALE = 2.0 ** -16
_MAX_SCALE = 2.0 ** +16
_PATIENCE = 2000


def _apply(xs, fn):
    if isinstance(xs, torch.Tensor):
        return fn(xs)
    if isinstance(xs, (str, bytes, np.ndarray)):
        return xs
    if isinstance(xs, abc.Mapping):
        return dict(_apply(kv, fn) for kv in xs.items())  # type: ignore
    if isinstance(xs, abc.Iterable):
        return type(xs)(_apply(x, fn) for x in xs)  # type: ignore
    return xs


def deep_to(xs, device: torch.device = None, dtype: torch.dtype = None):
    def fun(x):
        if x.is_floating_point():
            return x.to(device, dtype, non_blocking=True)
        return x.to(device, non_blocking=True)

    return _apply(xs, fun)


class OptContext:
    def __init__(self, optim: torch.optim.Optimizer):  # type: ignore
        self.optim = optim
        self.optim.load_state_dict(self.optim.state_dict())

    def zero_grad(self) -> None:
        self.optim.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self) -> None:
        self.optim.step()

    def state_dict(self) -> dict:
        return {
            k: (v if k != 'optim' else v.state_dict())
            for k, v in self.__dict__.items()
        }

    def load_state_dict(self, state_dict):
        state_dict = {**state_dict}
        self.optim.load_state_dict(state_dict.pop('optim'))

        dst = []
        _apply({k: self.__dict__[k] for k in state_dict}, dst.append)

        with torch.no_grad():
            self.__dict__.update(
                _apply(state_dict, lambda src: dst.pop(0).copy_(src)))

    def __enter__(self) -> 'OptContext':
        self.zero_grad()
        return self

    def __exit__(self, type_, *_) -> None:
        if type_ is None:
            self.step()


class _AmpContext(OptContext):
    _pending_steps: int = 0

    def __init__(self,
                 optim: torch.optim.Optimizer,
                 retry_on_inf: bool = True):
        self._retry_on_inf = retry_on_inf

        self._step = torch.zeros(1).int()
        self._scale = torch.empty(1).fill_(_MAX_SCALE)

        self._devices: set[torch.device] = set()
        self._grads: dict[nn.Parameter, torch.Tensor] = {}
        for param_group in optim.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    self._grads[p] = torch.empty_like(p)
                    self._devices.add(p.device)

        super().__init__(optim)

    def zero_grad(self) -> None:
        self._pending_steps = 0

    def _do_grad_transfer_and_check(self) -> bool:
        inv_scale = self._scale.double().reciprocal().float()
        inv_scales, infs = (
            {dev: inv_scale.to(dev) for dev in self._devices},
            {dev: torch.zeros(1, device=dev) for dev in self._devices},
        )
        for p in self._grads:  # unscale grads
            if p.grad is None:
                continue
            torch._amp_non_finite_check_and_unscale_(  # type: ignore
                p.grad, infs[p.device], inv_scales[p.device])

        found_inf = torch.cat(
            [inf.to(self._scale.device) for inf in infs.values()]).sum()
        self._scale = torch._amp_update_scale(  # type: ignore
            self._step, self._scale, found_inf, 2.0, 0.5, _PATIENCE)
        self._scale.clamp_(_MIN_SCALE, _MAX_SCALE)
        if found_inf.item():
            return False

        for p, grad in self._grads.items():  # update buffer grads
            if p.grad is not None:
                if not self._pending_steps:
                    grad.copy_(p.grad)
                else:
                    grad.add_(p.grad)
        return True

    def step(self) -> None:
        if not self._pending_steps:
            return
        with torch.no_grad():
            for p, grad in self._grads.items():  # update master grads
                if p.grad is not None:
                    p.grad.copy_(grad)
        super().step()

    def backward(self, loss: torch.Tensor) -> None:
        self._step = self._step.to(loss.device, non_blocking=True)
        self._scale = self._scale.to(loss.device, non_blocking=True)

        while self._scale.item() > _MIN_SCALE:  # type: ignore
            for p in self._grads:  # zero local grads
                if p.grad is not None:
                    p.grad.detach_().zero_()

            # collect grads
            (loss * self._scale).backward(retain_graph=self._retry_on_inf)

            with torch.no_grad():  # transfer grads and check for inf/nan
                if self._do_grad_transfer_and_check():
                    self._pending_steps += 1
                    return
                if not self._retry_on_inf:
                    return

        raise OverflowError(f'Cannot decrease scale below {_MIN_SCALE}')


def _deep_to_hook(_, xs, device=None, dtype=None):
    return deep_to(xs, device, dtype)


def get_amp_context(net: nn.Module,
                    opt: torch.optim.Optimizer | None = None,
                    fp16: bool = False,
                    retry_on_inf: bool = True) -> OptContext | None:
    """Switch model and optimizer to mixed precision mode

    Parameters:
    - fp16 - enables fp16 mode.
    - retry_on_inf - if set, probe another scale for the same loss
      when backward() fills grads with NaN/Inf. Only for fp16 mode.

    Example:
    ```
    # Creates model and optimizer in default precision
    model = Net()
    optimizer = optim.SGD(model.parameters(), ...)

    # Before training loop
    ctx = gnn.get_amp_context(model, optimizer, fp16=True)

    for input, target in data:
        with torch.cuda.amp.autocast():
            output = model(input)
        with ctx:  # Enter resets grads, exit updates parameters
            loss = loss_fn(output, target)
            ctx.backward(loss)
    ```

    """
    dtype = torch.float32
    if fp16:
        if all(
                torch.cuda.get_device_capability(dev) < (7, 0)
                for dev in range(torch.cuda.device_count())):
            warnings.warn('Neither of devices support hardware FP16, '
                          "that's why don't expect speed improvement")
        dtype = torch.float16

    # patch inputs
    net.register_forward_pre_hook(partial(_deep_to_hook,
                                          dtype=dtype))  # type: ignore

    # patch normalization
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._NormBase):  # type: ignore
            if m.affine:
                m.float()

    if opt is None:
        return None
    return OptContext(opt) if not fp16 else _AmpContext(opt, retry_on_inf)


def _warning_on_one_line(message,
                         category,
                         filename,
                         lineno,
                         file=None,
                         line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'


warnings.formatwarning = _warning_on_one_line  # type: ignore
