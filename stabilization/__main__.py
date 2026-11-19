"""Run the reference, alignment or combined synthetic experiment."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch
import torch.distributed as dist

from .method import align, fit_svd
from .metrics import mean_cosine, score_error
from .runtime import rank_and_size, runtime, synchronize
from .synthetic import SyntheticConfig, make_embeddings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "align", "demo"), nargs="?", default="demo")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per process")
    parser.add_argument("--users", type=int)
    parser.add_argument("--items", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--angle", type=float, help="Rotation in degrees")
    parser.add_argument("--noise", type=float, help="Scale of independent additive noise")
    parser.add_argument("--rtol", type=float, help="Relative singular value cutoff")
    parser.add_argument("--reference", type=Path, default=Path("runs/reference.pt"))
    parser.add_argument("--output", type=Path, help="Optional metrics JSON file")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.output and args.mode != "demo" and args.output.resolve() == args.reference.resolve():
        parser.error("--output and --reference must use different files.")
    return args


def timed(device, operation, *args):
    if dist.is_initialized():
        dist.barrier()
    synchronize(device)
    start = perf_counter()
    result = operation(*args)
    synchronize(device)
    elapsed = torch.tensor(perf_counter() - start, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return result, elapsed.item()


@contextmanager
def artifact_writer(path: Path):
    """Replace the destination only after the complete artifact has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile(dir=path.parent, delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            yield temp
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def experiment_config(args):
    stored = None
    values = asdict(SyntheticConfig())
    if args.mode == "align":
        stored = torch.load(args.reference, map_location="cpu", weights_only=True)
        if stored["format_version"] != 1:
            raise ValueError("Unsupported reference file version.")
        values.update(stored["synthetic"])

    for key in values:
        supplied = getattr(args, key)
        if supplied is not None:
            if stored and key in ("users", "items", "dim", "seed") and supplied != values[key]:
                raise ValueError(f"--{key} must match the saved reference.")
            values[key] = supplied

    dtype_name = args.dtype or (stored["dtype"] if stored else "float64")
    if stored and dtype_name != stored["dtype"]:
        raise ValueError("--dtype must match the saved reference.")
    rtol = args.rtol
    if rtol is None:
        rtol = stored["rtol"] if stored else values["dim"] * torch.finfo(getattr(torch, dtype_name)).eps
    return SyntheticConfig(**values), dtype_name, rtol, stored


@torch.no_grad()
def run(args, device):
    rank, size = rank_and_size()
    config, dtype_name, rtol, stored = experiment_config(args)
    shard = {"rank": rank, "size": size, "dtype": getattr(torch, dtype_name), "device": device}
    report = {
        "mode": args.mode,
        "synthetic": asdict(config),
        "dtype": dtype_name,
        "device": args.device,
        "processes": size,
        "threads_per_process": args.threads,
        "rtol": rtol,
        "torch_version": str(torch.__version__),
        "backend": dist.get_backend() if size > 1 else None,
    }

    users0, items0 = make_embeddings(config, version=0, **shard)
    if stored:
        stable_users0 = users0 @ stored["user"].to(device)
        stable_items0 = items0 @ stored["item"].to(device)
    else:
        ref, fit_time = timed(device, fit_svd, users0, items0, rtol)
        (stable_users0, stable_items0), apply_time = timed(device, ref.apply, users0, items0)
        report["reference"] = {
            "effective_rank": ref.effective_rank,
            "fit_seconds": fit_time,
            "apply_seconds": apply_time,
            "scores": score_error(users0, items0, stable_users0, stable_items0),
        }
        if args.mode == "reference" and rank == 0:
            checkpoint = {
                "format_version": 1,
                "synthetic": asdict(config),
                "dtype": dtype_name,
                "rtol": rtol,
                "user": ref.user.cpu(),
                "item": ref.item.cpu(),
            }
            with artifact_writer(args.reference) as output:
                torch.save(checkpoint, output)
            report["reference_file"] = str(args.reference)

    if args.mode in ("align", "demo"):
        users1, items1 = make_embeddings(config, version=1, **shard)
        current, fit_time = timed(device, fit_svd, users1, items1, rtol)
        aligned, align_time = timed(device, align, current, items1, stable_items0)
        (stable_users1, stable_items1), apply_time = timed(device, aligned.apply, users1, items1)
        report["iteration"] = {
            "effective_rank": aligned.effective_rank,
            "fit_seconds": fit_time,
            "align_seconds": align_time,
            "apply_seconds": apply_time,
            "scores": score_error(users1, items1, stable_users1, stable_items1),
            "cosine": {
                "users_before": mean_cosine(users0, users1),
                "users_after": mean_cosine(stable_users0, stable_users1),
                "items_before": mean_cosine(items0, items1),
                "items_after": mean_cosine(stable_items0, stable_items1),
            },
        }
    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    with runtime(args.device, args.threads) as device:
        report = run(args, device)
        if rank_and_size()[0] == 0:
            output = json.dumps(report, indent=2, allow_nan=False)
            if args.output:
                with artifact_writer(args.output) as artifact:
                    artifact.write((output + "\n").encode("utf-8"))
            print(output, flush=True)


if __name__ == "__main__":
    main()
