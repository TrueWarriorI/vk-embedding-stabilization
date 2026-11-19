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


def _rows(start, stop, config, stream, dtype, device):
    generator = np.random.PCG64(np.random.SeedSequence([config.seed, stream]))
    # A float64 uniform sample consumes one generator step.
    generator.advance(start * config.dim)
    values = np.random.Generator(generator).random((stop - start, config.dim))
    values *= 2.0 * math.sqrt(3.0)
    values -= math.sqrt(3.0)
    return torch.from_numpy(values).to(dtype=dtype, device=device)


def embedding_rows(
    config: SyntheticConfig,
    entity: str,
    start: int,
    stop: int,
    *,
    version: int,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate one contiguous block without materializing other rows."""
    if version not in (0, 1):
        raise ValueError("Synthetic data supports versions 0 and 1.")
    if entity not in ("users", "items") or not 0 <= start <= stop <= getattr(config, entity):
        raise ValueError("Invalid synthetic entity or row range.")
    stream = int(entity == "items")
    rows = _rows(start, stop, config, stream, dtype, device)
    if version == 1:
        if rotation is None:
            rotation = plane_rotation(config.dim, config.angle, dtype, device)
        rows = rows @ rotation
        if config.noise:
            rows += config.noise * _rows(start, stop, config, stream + 2, dtype, device)
    return rows


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
    rotation = plane_rotation(config.dim, config.angle, dtype, device) if version == 1 else None
    kwargs = {"version": version, "dtype": dtype, "device": device, "rotation": rotation}
    users = embedding_rows(config, "users", *row_bounds(config.users, rank, size), **kwargs)
    items = embedding_rows(config, "items", *row_bounds(config.items, rank, size), **kwargs)
    return users, items
