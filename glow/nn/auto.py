__all__ = ['Input', 'Model']

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Tuple, Type, Union

from torch import nn

from .. import countable
from ..api import get_wild_imports

_ARGS = frozenset({'input_size', 'num_features', 'in_features', 'in_channels'})


class ModuleWrapper(functools.partial):
    pass


@dataclass
class Input:
    channels: int
    module: nn.Module = nn.Module()
    leaves: List['Input'] = field(default_factory=list, init=False)

    def __or__(self, node: Union[ModuleWrapper, nn.Module]) -> 'Input':
        channels = self.channels

        module: nn.Module
        if isinstance(node, ModuleWrapper):
            if node.args and isinstance(node.args[0], int):
                channels = node.args[0]
            module = node.func(self.channels, *node.args, **node.keywords)

        elif isinstance(node, nn.Module):
            module = node

        else:
            raise TypeError(
                'Expected either ModuleWrapper, or torch.nn.Module class,' +
                f' got {type(node)}')

        leaf = Input(channels, module)
        self.leaves.append(leaf)
        return leaf


class Model(nn.ModuleDict):
    def __init__(self, inputs: List['Input'], outputs: List['Input']):
        super().__init__()

        count = countable()
        self.inputs = [f'{count(x)}' for x in inputs]

        roots = [*inputs]
        while roots:
            root = roots.pop()
            roots.extend(root.leaves)
            if all(leaf.module is None for leaf in root.leaves):
                continue
            self[f'{count(root)}'] = nn.ModuleDict(
                {f'{count(leaf)}': leaf.module for leaf in root.leaves})

        self.outputs = [f'{count(x)}' for x in outputs]

    def forward(self, *inputs):
        state = dict(zip(self.inputs, inputs))
        while {*self.outputs} - {*state}:
            for root_id in [*state]:
                if root_id not in self:
                    continue
                tensor = state.pop(root_id)
                state.update({
                    leaf_id: module(tensor)
                    for leaf_id, module in self[root_id].items()
                })

        return tuple(state[o] for o in self.outputs)


def _get_supported_modules() -> Iterator[Tuple[str, Any]]:
    for name in get_wild_imports(nn):
        module: Type[nn.Module] = getattr(nn, name)
        if not inspect.isclass(module) or not issubclass(module, nn.Module):
            continue

        sig = inspect.signature(module)
        if not sig.parameters:
            continue

        first, *rest = sig.parameters.values()
        if first.name not in _ARGS:
            continue

        wrapper = functools.partial(ModuleWrapper, module)
        wrapper.__signature__ = sig.replace(parameters=rest)  # type: ignore
        yield name, wrapper


for _name, _module in _get_supported_modules():
    globals()[_name] = _module
    __all__ += (_name, )  # type: ignore
