__all__ = ['plot_model']

import functools
from contextlib import ExitStack
from typing import Any, Dict, Tuple

import graphviz
import torch
from torch import nn
from torch.autograd import Function

from .. import Si, mangle


def id_(x: Any) -> str:
    if hasattr(x, 'variable'):
        x = x.variable
    return hex(x.storage().data_ptr() if torch.is_tensor(x) else id(x))


def as_tuple(xs):
    if xs is None:
        return ()
    if isinstance(xs, tuple):
        return tuple(x for x in xs if x is not None)
    return (xs, )


def sized(var: torch.Tensor):
    if max(var.shape) == var.numel():
        return f'{tuple(var.shape)}'
    return f'{tuple(var.shape)}\n{Si(var.numel())}'


class Builder:
    flat = False

    def __init__(self, inputs: set, params: dict):
        self.inputs = inputs
        self.params = params

        self._mangle = mangle()
        self._seen: Dict[str, str] = {}
        self._shapes: Dict[Function, str] = {}
        root = graphviz.Digraph(
            name='root',
            graph_attr={
                'rankdir': 'LR',
                'newrank': 'true',
                'color': 'lightgrey',
            },
            edge_attr={
                'labelfloat': 'true',
            },
            node_attr={
                'shape': 'box',
                'style': 'filled',
                'fillcolor': 'lightgrey',
                'fontsize': '12',
                'height': '0.2',
                'ranksep': '0.1',
            },
        )
        self.stack = [root]

    def _add_node(self, grad):
        root = self.stack[-1]
        # doesn't have variable, so it's "operation"
        if not hasattr(grad, 'variable'):
            label = type(grad).__name__.replace('Backward', '')
            if grad in self._shapes:
                label += f'\n=> {tuple(self._shapes[grad])}'
            root.node(id_(grad), label)
            return False

        # have variable, so it's either Parameter or Variable
        var, label = grad.variable, ''
        try:
            name = self.params[id(var)] + '\n'
            label = (name.partition if self.flat else name.rpartition)('.')[-1]
        except KeyError:
            root = self.stack[0]  # unnamed, that's why external
        label += f'{var.storage().data_ptr():x}\n'

        color = 'yellow' if id(var) in self.inputs else 'lightblue'
        root.node(id_(grad), label + sized(var), fillcolor=color)
        return True

    def _traverse_saved(self, grad):
        saved_tensors = grad.saved_tensors or ()
        saved_tensors = [var for var in saved_tensors if torch.is_tensor(var)]
        if not saved_tensors:
            return
        with self.stack[-1].subgraph() as s:
            s.attr(rank='same')
            for var in saved_tensors:
                label = hex(var.storage().data_ptr()) + '\n' + sized(var)
                # label = sized(var)
                if id_(var) not in self._seen:
                    s.node(id_(var), label, fillcolor='orange')
                s.edge(id_(var), id_(grad))

    def _traverse(self, grad, depth=0):
        if grad is None or id_(grad) in self._seen:
            return

        root = self.stack[-1]
        self._seen[id_(grad)] = head = root.name
        if self._add_node(grad):
            yield (depth - 1, None, grad)
            return

        # TODO : add merging of tensors with same data
        if hasattr(grad, 'saved_tensors'):
            self._traverse_saved(grad)

        for ch, _ in getattr(grad, 'next_functions', ()):
            if ch is None:
                continue
            yield from self._traverse(ch, depth + 1)

            tail = self._seen.get(id_(ch))
            if tail is not None and not (head.startswith(tail) or
                                         tail.startswith(head)):
                yield (depth, ch, grad)  # leafs, yield for depth-check
                continue

            name = self.params.get(id(getattr(ch, 'variable', None)))
            if not self.flat and name and name.rpartition('.')[0] == head:
                with root.subgraph() as s:
                    s.attr(rank='same')
                    s.edge(id_(ch), id_(grad))  # same module, same rank
            else:
                self.stack[0].edge(id_(ch), id_(grad))

    def _mark(self, ts):
        edges = []
        for t in as_tuple(ts):
            if t.grad_fn is not None:
                self._shapes[t.grad_fn] = t.shape
                edges.extend(self._traverse(t.grad_fn))
        if not edges:
            return

        max_depth = max(depth for depth, *_ in edges) + 1
        for depth, tail, head in edges:  # inter-module edges
            if tail is not None:
                minlen = None if self.flat else f'{max_depth - depth}'
                self.stack[0].edge(id_(tail), id_(head), minlen=minlen)

    def forward_pre(self, name, module, xs):
        self._mark(xs)
        # -------- start node --------
        if self.flat:
            return
        scope = graphviz.Digraph(name=self._mangle(name))
        scope.attr(label=f'{name.split(".")[-1]}:{type(module).__name__}')
        self.stack.append(scope)

    def forward(self, module, _, ys):
        self._mark(ys)
        if self.flat:
            return
        cluster = self.stack.pop(-1)
        cluster.name = f'cluster_{cluster.name}'
        self.stack[-1].subgraph(cluster)
        # -------- end node --------


def plot_model(model: nn.Module, *input_shapes: Tuple[int, ...], device='cpu'):
    """Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    """
    inputs = [torch.zeros(1, *shape, device=device) for shape in input_shapes]
    inputs = [inp.requires_grad_() for inp in inputs]
    params = model.state_dict(prefix='root.', keep_vars=True)
    hk = Builder(
        {id(var) for var in inputs},
        {id(var): name for name, var in params.items()},
    )
    with ExitStack() as stack:
        for name, m in model.named_modules(prefix='root'):
            stack.callback(
                m.register_forward_pre_hook(
                    functools.partial(hk.forward_pre, name)).remove)
            stack.callback(m.register_forward_hook(hk.forward).remove)
        model(*inputs)

    dot = hk.stack.pop()
    assert not hk.stack

    dot.filename = getattr(model, 'name', type(model).__qualname__)
    dot.directory = 'graphs'
    dot.format = 'svg'

    size_min = 12
    scale_factor = .15
    size = max(size_min, len(dot.body) * scale_factor)

    dot.graph_attr.update(size=f'{size},{size}')
    dot.render(cleanup=True)
    return dot
