"""Process group setup shared by the command-line experiments."""

import os
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist


def validate_world_size(size: int) -> None:
    if size < 1 or size & (size - 1):
        raise ValueError("TSQR requires a power-of-two process count (1, 2, 4, 8, ...).")


@contextmanager
def runtime(device_name: str, threads: int):
    size = int(os.environ.get("WORLD_SIZE", "1"))
    validate_world_size(size)
    torch.set_num_threads(threads)
    device = torch.device(device_name)
    backend = "gloo"
    if device_name == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise ValueError("CUDA requires one visible GPU per local process.")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
        if size > 1 and not dist.is_nccl_available():
            raise ValueError("Multi-process CUDA requires a PyTorch build with NCCL.")

    if size > 1:
        dist.init_process_group(backend, timeout=timedelta(seconds=90))
    try:
        yield device
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def rank_and_size() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
