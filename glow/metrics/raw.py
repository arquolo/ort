__all__ = ['accuracy_', 'auroc', 'average_precision', 'dice']

import torch

from .base import to_index, to_prob


def accuracy_(pred, true) -> torch.Tensor:
    _, pred, true = to_index(pred, true)
    return (true == pred).double().mean()


def dice(pred, true, macro=False) -> torch.Tensor:
    c, pred, true = to_index(pred, true)

    def _apply(pred, true) -> torch.Tensor:
        true = true.view(-1)
        pred = pred.view(-1)
        tp, t, p = (
            x.bincount(minlength=c).clamp_(1).double()
            for x in (true[true == pred], true, pred))
        return 2 * tp / (t + p)

    if macro:
        return _apply(pred, true)

    b = pred.shape[0]
    out = map(_apply, pred.view(b, -1).unbind(), true.view(b, -1).unbind())
    return torch.mean(torch.stack([*out]), dim=0)


def _rankdata(ten: torch.Tensor) -> torch.Tensor:
    sorter = ten.argsort()
    ten = ten[sorter]

    diff = torch.cat([torch.tensor([True]), ten[1:] != ten[:-1]])
    # diff = np.r_[True, ten[1:] != ten[:-1]]

    dense = diff.cumsum(0)[sorter.argsort()]

    diff = diff.nonzero(as_tuple=False).view(-1)
    count = torch.cat([diff, torch.tensor([diff.numel()])])
    # count = np.r_[diff.nonzero(diff).view(-1), diff.numel()]

    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _binary_metric(fn):
    def call(pred, true, index: int = 0) -> torch.Tensor:
        c, pred, true = to_prob(pred, true)
        assert 0 <= index < c
        return fn(pred[:, index].view(-1), (true == index).view(-1))

    return call


@_binary_metric
def auroc(pred, true) -> torch.Tensor:
    n = true.numel()
    n_pos = true.sum()

    r = _rankdata(pred)
    total = n_pos * (n - n_pos)
    return (r[true == 1].sum() - n_pos * (n_pos + 1) // 2) / float(total)


@_binary_metric
def average_precision(pred, true) -> torch.Tensor:
    n = true.numel()
    n_pos = true.sum()

    true = true[torch.argsort(pred)].flipud()
    weights = torch.arange(1, n + 1).float().reciprocal()
    precision = true.cumsum(0).float()
    return torch.einsum('i,i,i', true.float(), precision, weights) / n_pos
