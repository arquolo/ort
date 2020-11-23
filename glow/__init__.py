# flake8: noqa
"""Collection of tools for easier prototyping with deep learning extensions"""

"""TODO:
Add:
    glow.buffered
    - Offloading to other process. Use loky as backend.

    glow.mapped
    - Proper serialization of np.ndarray/np.memmap
      - `parent` -> `child` = move to shared memory if size allows
      - `child` -> `parent` = keep in shared if already there,
        otherwise move as usual.
      - Drop shared data at pool shutdown.

    glow.{nn.make_loader -> utils.make_loader}
    - IterableDataset loader. Use chunked + roundrobin + buffered as baseline.
    - Seed as argument to toggle patching of dataset and iterable
      to provide batchsize- and workers-invariant data generation

    glow.{api -> env}
    - `env.get(name, default)` - get value from current env
    - `env.register(name, **aliases: object)` - add aliases for name
    - `env.set(name, value)` - set value to env
    - `env.fork() -> ContextManager` - temporarily fork env

Docs:
    Add for all exported functions

Refactor:
    glow.{nn.plot -> utils.plot}
    - Fix to make it working

    glow.__init__
    - Add explicit imports from glow.core.*

    glow.io._tiled
    - Make external factory to combine all 3 classes with memoization. Drop
      usage of metaclass as ambiguous.

    glow.nn.modules
    - Drop garbage, redesign all of it. Use glow.env as storage for options.

Chore:
    Remove wrapt as dependency (used for decorator, ObjectProxy)
"""


from . import core
from .core import *

__all__ = core.__all__
