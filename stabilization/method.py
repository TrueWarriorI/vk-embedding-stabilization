"""Low-rank score SVD and orthogonal alignment."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .runtime import rank_and_size
from .tsqr import tsqr_r


@dataclass(frozen=True)
class Transforms:
    user: torch.Tensor
    item: torch.Tensor
    singular_values: torch.Tensor
    effective_rank: int

    def apply(self, users: torch.Tensor, items: torch.Tensor):
        return users @ self.user, items @ self.item


def fit_svd(
    users: torch.Tensor, items: torch.Tensor, rtol: float | None = None
) -> Transforms:
    """Fit score-preserving transforms from local row shards."""
    if users.ndim != 2 or items.ndim != 2 or users.shape[1] != items.shape[1]:
        raise ValueError("User and item embeddings must have the same feature width.")
    if users.device != items.device or users.dtype != items.dtype:
        raise ValueError("User and item embeddings must share a device and dtype.")
    dim = users.shape[1]
    if dim == 0 or users.dtype not in (torch.float32, torch.float64):
        raise ValueError("Use positive-width float32 or float64 embeddings.")
    if rtol is None:
        rtol = dim * torch.finfo(users.dtype).eps
    if not 0 <= rtol < 1:
        raise ValueError("rtol must be in [0, 1).")

    r_item = tsqr_r(items)
    r_user = tsqr_r(users)
    rank, size = rank_and_size()
    if rank == 0:
        left, singular, right_t = torch.linalg.svd(r_item @ r_user.T)
        keep = singular > singular[0] * rtol
        safe_singular = singular.masked_fill(~keep, 1)
        weights = safe_singular.rsqrt() * keep
        user_transform = (r_item.T @ left) * weights
        item_transform = (r_user.T @ right_t.T) * weights
        effective_rank = keep.sum()
    else:
        user_transform = users.new_empty((dim, dim))
        item_transform = items.new_empty((dim, dim))
        singular = users.new_empty(dim)
        effective_rank = users.new_empty((), dtype=torch.int64)

    if size > 1:
        for tensor in (user_transform, item_transform, singular, effective_rank):
            dist.broadcast(tensor, src=0)
    return Transforms(user_transform, item_transform, singular, int(effective_rank.item()))


def procrustes(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Map row embeddings to reference using matching item IDs on each rank."""
    if current.ndim != 2 or current.shape != reference.shape:
        raise ValueError("Align the same item IDs in the same row order on each rank.")
    cross = current.T @ reference
    rank, size = rank_and_size()
    if size > 1:
        dist.reduce(cross, dst=0)
    if rank == 0:
        left, _, right_t = torch.linalg.svd(cross)
        rotation = left @ right_t
    else:
        rotation = torch.empty_like(cross)
    if size > 1:
        dist.broadcast(rotation, src=0)
    return rotation


def align(
    transforms: Transforms, items: torch.Tensor, reference_items: torch.Tensor
) -> Transforms:
    rotation = procrustes(items @ transforms.item, reference_items)
    return Transforms(
        transforms.user @ rotation,
        transforms.item @ rotation,
        transforms.singular_values,
        transforms.effective_rank,
    )
