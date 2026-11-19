"""Fit and apply stabilization to memory-mapped NumPy matrices."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist

from .artifacts import write_report
from .method import fit_svd_from_r, procrustes_from_cross
from .runtime import rank_and_size, runtime
from .synthetic import SyntheticConfig, embedding_rows, plane_rotation, row_bounds
from .tsqr import local_r, merge_r


@dataclass(frozen=True)
class Rows:
    values: np.ndarray
    ids: np.ndarray

    @classmethod
    def open(cls, folder: Path, entity: str) -> "Rows":
        values = np.load(folder / f"{entity}.npy", mmap_mode="r", allow_pickle=False)
        ids = np.load(folder / f"{entity[:-1]}_ids.npy", mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or min(values.shape) < 1 or values.dtype not in (np.float32, np.float64):
            raise ValueError(f"{entity}.npy must contain a nonempty float32 or float64 matrix.")
        if ids.shape != (len(values),) or ids.dtype != np.int64:
            raise ValueError(f"Provide one int64 ID per row in {entity[:-1]}_ids.npy.")
        if not ids.flags.c_contiguous:
            raise ValueError("ID arrays must be contiguous.")
        return cls(values, ids)

    def blocks(self, block_rows: int):
        rank, size = rank_and_size()
        first, last = row_bounds(len(self.ids), rank, size)
        for start in range(first, last, block_rows):
            stop = min(start + block_rows, last)
            yield slice(start, stop)

    def read(self, indices) -> torch.Tensor:
        return torch.from_numpy(np.array(self.values[indices], copy=True, order="C"))

    def check_ids(self, interval: slice) -> None:
        start = max(0, interval.start - 1)
        ids = self.ids[start:interval.stop]
        if np.any(ids[1:] <= ids[:-1]):
            raise ValueError("IDs must be strictly increasing and unique.")


def coordinated(operation):
    """Propagate I/O or input errors to every rank before the next collective."""
    _, size = rank_and_size()
    if size == 1:
        return operation()
    result = None
    error = None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * size
    dist.all_gather_object(errors, error)
    if any(errors):
        details = "; ".join(f"rank {rank}: {err}" for rank, err in enumerate(errors) if err)
        raise RuntimeError(details)
    return result


@contextmanager
def output_directory(destination: Path):
    """Publish complete output in a single rename after all workers finish."""
    rank, size = rank_and_size()
    destination = destination.resolve()
    staging = None

    def create():
        nonlocal staging
        if rank == 0:
            if destination.exists():
                raise FileExistsError(f"Output already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            folder = destination.with_name(f".{destination.name}-{uuid4().hex}")
            # Inherit the output parent's permissions, including on Windows.
            folder.mkdir()
            staging = folder
            return str(folder)

    try:
        folders = [coordinated(create)]
        if size > 1:
            dist.broadcast_object_list(folders, src=0)
        folder = Path(folders[0])
        yield folder
        coordinated(lambda: folder.rename(destination) if rank == 0 else None)
    finally:
        if staging is not None and staging.exists():
            if staging.resolve() != staging or staging.parent != destination.parent:
                raise RuntimeError(f"Unexpected staging directory: {staging}")
            rmtree(staging)


def create_array(path: Path, dtype, shape) -> None:
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array.flush()


@torch.no_grad()
def generate(destination: Path, config: SyntheticConfig, dtype, block_rows: int) -> None:
    if block_rows < 1 or dtype not in (torch.float32, torch.float64):
        raise ValueError("Use a positive block size and float32 or float64.")
    if rank_and_size()[1] > 1:
        raise ValueError("Run generate as a single process.")
    numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
    rotation = plane_rotation(config.dim, config.angle, dtype, "cpu")
    with output_directory(destination) as staging:
        for version in (0, 1):
            folder = staging / f"version{version}"
            folder.mkdir()
            for entity in ("users", "items"):
                count = getattr(config, entity)
                path = folder / f"{entity}.npy"
                ids_path = folder / f"{entity[:-1]}_ids.npy"
                matrix = np.lib.format.open_memmap(
                    path, mode="w+", dtype=numpy_dtype, shape=(count, config.dim)
                )
                ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.int64, shape=(count,))
                for start in range(0, count, block_rows):
                    stop = min(start + block_rows, count)
                    matrix[start:stop] = embedding_rows(
                        config, entity, start, stop, version=version, dtype=dtype, rotation=rotation
                    ).numpy()
                    ids[start:stop] = np.arange(start, stop, dtype=np.int64)
                matrix.flush()
                ids.flush()
                del matrix, ids


def scan_users(users: Rows, block_rows: int) -> torch.Tensor:
    factor = local_r(users.read(slice(0, 0)))
    for interval in users.blocks(block_rows):
        users.check_ids(interval)
        factor = merge_r(factor, local_r(users.read(interval)))
    return factor


def scan_items(items: Rows, block_rows: int, reference=None):
    factor = local_r(items.read(slice(0, 0)))
    cross = torch.zeros_like(factor)
    anchors = 0
    for interval in items.blocks(block_rows):
        items.check_ids(interval)
        values = items.read(interval)
        factor = merge_r(factor, local_r(values))
        if reference is not None:
            ref_items, ref_transform = reference
            ids = items.ids[interval]
            positions = np.searchsorted(ref_items.ids, ids)
            matched = np.flatnonzero(positions < len(ref_items.ids))
            matched = matched[ref_items.ids[positions[matched]] == ids[matched]]
            if matched.size:
                ref_values = ref_items.read(positions[matched]) @ ref_transform
                cross += values[matched].T @ ref_values
                anchors += len(matched)
    return factor, cross, anchors


def open_inputs(folder: Path, reference_folder, checkpoint_path, block_rows: int):
    users, items = Rows.open(folder, "users"), Rows.open(folder, "items")
    if users.values.shape[1] != items.values.shape[1] or users.values.dtype != items.values.dtype:
        raise ValueError("User and item matrices must share the same dtype and width.")
    reference = None
    if reference_folder is not None:
        ref_items = Rows.open(reference_folder, "items")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        factor = checkpoint.get("item")
        dim = items.values.shape[1]
        if (
            ref_items.values.shape[1] != dim
            or ref_items.values.dtype != items.values.dtype
            or not isinstance(factor, torch.Tensor)
            or factor.shape != (dim, dim)
            or factor.dtype != items.read(slice(0, 0)).dtype
            or checkpoint.get("kind") != "streaming-reference-v1"
            or checkpoint.get("reference_items") != len(ref_items.ids)
        ):
            raise ValueError("Reference data and checkpoint must come from a compatible reference run.")
        for interval in ref_items.blocks(block_rows):
            ref_items.check_ids(interval)
        reference = ref_items, factor
    return users, items, reference


def initialize_output(folder: Path, datasets):
    if rank_and_size()[0] == 0:
        for name, rows in datasets:
            create_array(folder / f"{name}.npy", rows.values.dtype, rows.values.shape)
            create_array(folder / f"{name[:-1]}_ids.npy", np.int64, rows.ids.shape)


def write_transformed(folder: Path, datasets, transforms, block_rows: int) -> None:
    for (name, rows), transform in zip(datasets, (transforms.user, transforms.item), strict=True):
        values = np.load(folder / f"{name}.npy", mmap_mode="r+", allow_pickle=False)
        ids = np.load(folder / f"{name[:-1]}_ids.npy", mmap_mode="r+", allow_pickle=False)
        for interval in rows.blocks(block_rows):
            values[interval] = (rows.read(interval) @ transform).numpy()
            ids[interval] = rows.ids[interval]
        values.flush()
        ids.flush()


@torch.no_grad()
def transform_files(
    folder: Path, destination: Path, block_rows: int, *, reference_folder=None, checkpoint_path=None
) -> dict:
    if block_rows < 1 or (reference_folder is None) != (checkpoint_path is None):
        raise ValueError("Provide positive block size and both or neither reference arguments.")
    users, items, reference = coordinated(
        lambda: open_inputs(folder, reference_folder, checkpoint_path, block_rows)
    )
    user_r, (item_r, cross, anchors) = coordinated(
        lambda: (scan_users(users, block_rows), scan_items(items, block_rows, reference))
    )
    transforms = fit_svd_from_r(user_r, item_r)
    rank, size = rank_and_size()
    if reference is not None:
        total = torch.tensor(anchors, dtype=torch.int64)
        if size > 1:
            dist.all_reduce(total)
        anchors = total.item()
        if anchors == 0:
            raise ValueError("No item IDs overlap with the reference.")
        # (items @ M).T @ reference == M.T @ (items.T @ reference).
        transforms = transforms.rotated(procrustes_from_cross(transforms.item.T @ cross))

    datasets = (("users", users), ("items", items))
    report = {
        "mode": "align" if reference is not None else "reference",
        "users": len(users.ids),
        "items": len(items.ids),
        "dim": users.values.shape[1],
        "dtype": str(users.values.dtype),
        "block_rows": block_rows,
        "processes": size,
        "effective_rank": transforms.effective_rank,
        "anchor_items": anchors if reference is not None else None,
    }
    with output_directory(destination) as staging:
        coordinated(lambda: initialize_output(staging, datasets))
        coordinated(lambda: write_transformed(staging, datasets, transforms, block_rows))

        def save():
            if rank == 0:
                torch.save(
                    {
                        "kind": "streaming-aligned-v1" if reference is not None else "streaming-reference-v1",
                        "user": transforms.user,
                        "item": transforms.item,
                        "reference_items": len(items.ids),
                    },
                    staging / "transforms.pt",
                )
                write_report(report, staging / "metrics.json")

        coordinated(save)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    generator = modes.add_parser("generate", help="Write two synthetic versions to disk")
    generator.add_argument("--users", type=int, default=4096)
    generator.add_argument("--items", type=int, default=3072)
    generator.add_argument("--dim", type=int, default=32)
    generator.add_argument("--seed", type=int, default=42)
    generator.add_argument("--angle", type=float, default=75)
    generator.add_argument("--noise", type=float, default=0)
    generator.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    reference = modes.add_parser("reference", help="Fit and save a reference version")
    align = modes.add_parser("align", help="Align a new version using overlapping item IDs")
    align.add_argument("--reference", type=Path, required=True)
    align.add_argument("--checkpoint", type=Path, required=True)
    for mode in (reference, align):
        mode.add_argument("--input", type=Path, required=True)
    for mode in (generator, reference, align):
        mode.add_argument("--output", type=Path, required=True)
        mode.add_argument("--block-rows", type=int, default=1024)
        mode.add_argument("--threads", type=int, default=1)

    args = parser.parse_args()
    if args.block_rows < 1 or args.threads < 1:
        parser.error("--block-rows and --threads must be positive.")
    with runtime("cpu", args.threads):
        if args.mode == "generate":
            config = SyntheticConfig(args.users, args.items, args.dim, args.seed, args.angle, args.noise)
            generate(args.output, config, getattr(torch, args.dtype), args.block_rows)
            print(args.output.resolve())
        else:
            transform_files(
                args.input, args.output, args.block_rows,
                reference_folder=getattr(args, "reference", None),
                checkpoint_path=getattr(args, "checkpoint", None),
            )


if __name__ == "__main__":
    main()
