"""R-only tall-skinny QR with a balanced binary reduction tree."""

import torch
import torch.distributed as dist

from .runtime import rank_and_size, validate_world_size


def local_r(matrix: torch.Tensor) -> torch.Tensor:
    """Return a square R, padding short or empty row blocks with zeros."""
    width = matrix.shape[1]
    if matrix.shape[0] == 0:
        return matrix.new_zeros((width, width))
    factor = torch.linalg.qr(matrix, mode="r").R
    return torch.nn.functional.pad(factor, (0, 0, 0, width - factor.shape[0]))


def tsqr_r(matrix: torch.Tensor) -> torch.Tensor | None:
    """Return the global R on rank zero; other ranks return None."""
    rank, size = rank_and_size()
    validate_world_size(size)
    factor = local_r(matrix)
    step = 1
    while step < size:
        if rank % (2 * step) == step:
            dist.send(factor.contiguous(), dst=rank - step)
            return None

        peer = torch.empty_like(factor, memory_format=torch.contiguous_format)
        dist.recv(peer, src=rank + step)
        factor = local_r(torch.cat((factor, peer), dim=0))
        step *= 2
    return factor
