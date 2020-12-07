__all__ = [
    'device', 'dump_to_onnx', 'frozen', 'inference', 'param_count', 'profile'
]

import functools
from contextlib import ExitStack, contextmanager
from io import BytesIO
from typing import Callable, Iterator, Tuple, TypeVar, cast

import torch
import torch.autograd
import torch.cuda
import torch.onnx
from torch import nn

from .. import Si

_F = TypeVar('_F', bound=Callable[..., Iterator])


def device() -> torch.device:
    """Gets current device, including CPU"""
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    return torch.device('cpu')


@contextmanager
def _set_eval(net: nn.Module) -> Iterator[None]:
    """Locally switch net to eval-mode"""
    was_train = net.training
    try:
        net.eval()
        yield
    finally:
        net.train(was_train)


@contextmanager
def frozen(net: nn.Module) -> Iterator[None]:
    """Blocks net from changing its state. Useful while training.

    Net is switched to eval mode, and its parameters are detached from graph.
    Grads are computed if inputs require grad.
    Works as context manager"""
    with ExitStack() as stack:
        stack.enter_context(_set_eval(net))
        for p in net.parameters():
            if p.requires_grad:
                stack.callback(p.requires_grad_)
                p.detach_()
        yield


@contextmanager
def inference(net: nn.Module) -> Iterator[None]:
    """Blocks net from changing its state. Useful while inference.

    Net is switched to eval mode, and gradient computation is turned off.
    Works as context manager"""
    with _set_eval(net):
        with torch.no_grad():
            yield


def param_count(net: nn.Module) -> Si:
    """Count of parameters in net, both training and not"""
    return Si(sum(p.numel() for p in net.parameters()))


def profile(fn: _F) -> _F:
    """Decorator to profile CUDA ops. Use with `nvprof`

    Use in script launched via:
    ```bash
    nvprof --profile-from-start off -o trace.prof -- python main.py
    ```
    Usage:
    >>> @profile
    ... def train_loop():
    ...     for data in loader:
    ...         yield step(data)

    """
    def wrapper(*args, **kwargs):
        results = fn(*args, **kwargs)
        with torch.cuda.profiler.profile():
            yield next(results)
            with torch.autograd.profiler.emit_nvtx():
                yield from results

    return cast(_F, functools.update_wrapper(wrapper, fn))


def dump_to_onnx(net: nn.Module, *shapes: Tuple[int, ...],
                 device='cpu') -> bytes:
    """Converts model to ONNX graph, represented as bytes

    Parameters:
    - net - torch.nn.Module to convert
    - shapes - Shapes of input data, all except batch dimension

    Example usage:
    >>> net = torch.nn.Linear(4, 4)
    >>> bytes_ = dump_to_onnx(net, [4])

    To restore graph:
    >>> from onnxruntime import backend
    >>> rep = backend.prepare(bytes_or_filename, device='cpu')
    >>> rep.run([np.zeros(4, 4)])[0]

    """
    dynamic_axes = {
        f'inp_{i}': {
            0: 'batch',
            **{dim: f'inp_{i}_dim_{dim}' for dim in range(2, 1 + len(shape))}
        } for i, shape in enumerate(shapes)
    }
    buf = BytesIO()
    torch.onnx.export(
        net.to(device).eval(),
        tuple(
            torch.rand(1, *shape, requires_grad=True, device=device)
            for shape in shapes),
        buf,
        input_names=[*dynamic_axes],
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True)
    return buf.getvalue()
