from __future__ import annotations  # until 3.10

__all__ = ['tiramisu']

import itertools

from torch import nn

from .. import windowed
from .modules_factory import Cat, DenseBlock, conv
from .util import param_count


def tiramisu(cin, cout, init=48, depths=(4, 4), step=12):
    steps = itertools.accumulate([init] + [step * depth for depth in depths])
    steps = [*steps]

    core: list[nn.Module] = [
        DenseBlock(steps[-2], depth=depths[-1], step=step),
    ]
    slides = windowed(steps, size=3)
    f_out = depth = 0
    for i, (f_in, f_middle, f_out) in reversed([*enumerate(slides)]):
        f_delta = f_out - f_middle
        depth = (f_middle - f_in) // step
        is_last = not i
        core = [
            conv(f_middle, padding=0, order='NA-'),
            nn.MaxPool2d(2),
            *core,
            nn.ConvTranspose2d(f_delta, f_delta, 2, stride=2),
        ]
        core = [
            DenseBlock(f_in, depth=depth, step=step, full=True),
            Cat(*core),
            DenseBlock(f_out, depth=depth, step=step, full=is_last),
        ]

    core = [
        nn.Conv2d(cin, init, 3, padding=1),
        *core,
        nn.Conv2d(f_out + depth * step, cout, 1),
    ]
    model = nn.Sequential(*core)
    layers = sum(isinstance(m, nn.Conv2d) for m in model.modules())
    model.name = f'tiramisu_{layers}_{param_count(model)}'
    return model
