"""Distributed verification without building the full score matrix."""

import math

import torch
import torch.distributed as dist

from .runtime import rank_and_size


def _sum(tensor: torch.Tensor) -> torch.Tensor:
    if rank_and_size()[1] > 1:
        dist.all_reduce(tensor)
    return tensor


def mean_cosine(first: torch.Tensor, second: torch.Tensor) -> dict:
    """Mean cosine over matching rows with two nonzero norms."""
    first = first.to(torch.float64)
    second = second.to(torch.float64)
    norms = torch.linalg.vector_norm(first, dim=1) * torch.linalg.vector_norm(second, dim=1)
    valid = norms > 0
    safe_norms = norms.masked_fill(~valid, 1)
    cosine = ((first * second).sum(dim=1) / safe_norms).clamp(-1, 1)
    result = _sum(torch.stack(((cosine * valid).sum(), valid.sum())))
    total, count = result.tolist()
    return {"mean": total / count if count else None, "count": int(count)}


def score_error(
    users: torch.Tensor,
    items: torch.Tensor,
    stable_users: torch.Tensor,
    stable_items: torch.Tensor,
    samples_per_rank: int = 1024,
) -> dict:
    """Sample user-item pairs in each shard and aggregate error across ranks."""
    count = min(samples_per_rank, len(users) * len(items))
    index = torch.arange(count, device=users.device)
    if count:
        pairs = index * (len(users) * len(items)) // count
        user_idx = pairs // len(items)
        item_idx = (pairs + user_idx) % len(items)
        before = (users[user_idx] * items[item_idx]).sum(dim=1).double()
        after = (stable_users[user_idx] * stable_items[item_idx]).sum(dim=1).double()
        error = after - before
        max_error = error.abs().max()
    else:
        before = users.new_empty(0, dtype=torch.float64)
        error = before
        max_error = users.new_zeros((), dtype=torch.float64)

    stats = torch.stack((error.square().sum(), before.square().sum(), before.new_tensor(count)))
    squared_error, squared_score, total = _sum(stats).tolist()
    if rank_and_size()[1] > 1:
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
    return {
        "pairs": int(total),
        "relative_l2": math.sqrt(squared_error / squared_score) if squared_score else None,
        "max_abs": max_error.item() if total else None,
    }
