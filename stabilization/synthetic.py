"""Reproducible synthetic row shards with an exact rotation and optional drift."""

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class SyntheticConfig:
    users: int = 4096
    items: int = 3072
    dim: int = 32
    seed: int = 42
    angle: float = 75.0
    noise: float = 0.0

    def __post_init__(self):
        if self.users < 1 or self.items < 1 or self.dim < 2 or self.seed < 0:
            raise ValueError("Provide positive row counts, dim >= 2 and seed >= 0.")
        if not math.isfinite(self.angle) or not math.isfinite(self.noise) or self.noise < 0:
            raise ValueError("The angle must be finite and noise must be finite and nonnegative.")


def row_bounds(total: int, rank: int, size: int) -> tuple[int, int]:
    return total * rank // size, total * (rank + 1) // size


def plane_rotation(dim: int, degrees: float, dtype, device) -> torch.Tensor:
    """Rotate consecutive coordinate planes by the same angle."""
    angle = math.radians(degrees)
    rotation = torch.eye(dim, dtype=dtype, device=device)
    first = torch.arange(0, dim - 1, 2, device=device)
    second = first + 1
    rotation[first, first] = math.cos(angle)
    rotation[second, second] = math.cos(angle)
    rotation[first, second] = math.sin(angle)
    rotation[second, first] = -math.sin(angle)
    return rotation


def _rows(total, config, rank, size, stream, dtype, device):
    start, stop = row_bounds(total, rank, size)
    generator = np.random.PCG64(np.random.SeedSequence([config.seed, stream]))
    # A float64 uniform sample consumes one generator step.
    generator.advance(start * config.dim)
    values = np.random.Generator(generator).random((stop - start, config.dim))
    values *= 2.0 * math.sqrt(3.0)
    values -= math.sqrt(3.0)
    return torch.from_numpy(values).to(dtype=dtype, device=device)


def make_embeddings(
    config: SyntheticConfig,
    *,
    version: int,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    rank: int = 0,
    size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (users, items). Row indices are IDs shared by both versions."""
    if version not in (0, 1):
        raise ValueError("Synthetic data supports versions 0 and 1.")
    users = _rows(config.users, config, rank, size, 0, dtype, device)
    items = _rows(config.items, config, rank, size, 1, dtype, device)
    if version == 0:
        return users, items

    rotation = plane_rotation(config.dim, config.angle, dtype, device)
    users = users @ rotation
    items = items @ rotation
    if config.noise:
        users += config.noise * _rows(config.users, config, rank, size, 2, dtype, device)
        items += config.noise * _rows(config.items, config, rank, size, 3, dtype, device)
    return users, items
