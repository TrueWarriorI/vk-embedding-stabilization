"""Benchmark CPU QR and TSQR with an equal total thread budget."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
import torch.distributed as dist

from .artifacts import write_report
from .launch import torchrun_command
from .runtime import rank_and_size, runtime, validate_world_size
from .synthetic import SyntheticConfig, embedding_rows, row_bounds
from .tsqr import tsqr_r


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("page_faults", wintypes.DWORD),
                ("peak_working_set", ctypes.c_size_t),
                ("other_counters", ctypes.c_size_t * 7),
            ]

        counters = MemoryCounters()
        counters.size = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.peak_working_set

    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


@torch.no_grad()
def measure_worker(args) -> None:
    rows = args.sizes[0]
    processes = args.processes[0]
    threads = args.total_threads // processes
    with runtime("cpu", threads):
        rank, size = rank_and_size()
        if size != processes or (args.worker == "qr" and size != 1):
            raise ValueError("The worker process count must match the launcher.")
        config = SyntheticConfig(users=rows, dim=args.dim, seed=args.seed)
        matrix = embedding_rows(config, "users", *row_bounds(rows, rank, size), version=0)

        def factorize():
            if args.worker == "qr":
                return torch.linalg.qr(matrix, mode="r").R
            return tsqr_r(matrix)

        for _ in range(args.warmups):
            factorize()
        samples = []
        for _ in range(args.repeats):
            if size > 1:
                dist.barrier()
            start = perf_counter()
            result = factorize()
            duration = torch.tensor(perf_counter() - start, dtype=torch.float64)
            if size > 1:
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            samples.append(duration.item())

        gram = matrix.T @ matrix
        peak = torch.tensor(peak_rss_bytes(), dtype=torch.int64)
        if size > 1:
            dist.reduce(gram, dst=0)
            dist.reduce(peak, dst=0, op=dist.ReduceOp.MAX)
        if rank == 0:
            error = torch.linalg.matrix_norm(result.T @ result - gram)
            relative_error = (error / torch.linalg.matrix_norm(gram)).item()
            q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
            write_report(
                {
                    "algorithm": args.worker,
                    "rows": rows,
                    "processes": size,
                    "threads_per_process": threads,
                    "qr_seconds": {
                        "median": statistics.median(samples),
                        "q1": q1,
                        "q3": q3,
                        "samples": samples,
                    },
                    "peak_rss_bytes_per_rank_max": peak.item(),
                    "relative_gram_error": relative_error,
                },
                args.output,
            )


def run_config(args, rows: int, processes: int, algorithm: str, path: Path) -> dict:
    module = "stabilization.benchmark"
    if processes > 1:
        command, env = torchrun_command(processes, module)
    else:
        command, env = [sys.executable, "-m", module], None
    command += [
        "--worker", algorithm,
        "--sizes", str(rows),
        "--processes", str(processes),
        "--dim", str(args.dim),
        "--total-threads", str(args.total_threads),
        "--warmups", str(args.warmups),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--output", str(path),
    ]
    print(f"Benchmark: {algorithm}, {rows} rows, {processes} processes", file=sys.stderr, flush=True)
    start = perf_counter()
    completed = subprocess.run(command, env=env, capture_output=True, check=False)
    wall_seconds = perf_counter() - start
    if completed.returncode:
        sys.stderr.buffer.write(completed.stdout + completed.stderr)
        raise RuntimeError(f"Worker command exited with code {completed.returncode}.")
    result = json.loads(path.read_text(encoding="utf-8"))
    result["command_wall_seconds"] = wall_seconds
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 65536])
    parser.add_argument("--processes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--total-threads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("qr", "tsqr"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        for count in args.processes:
            validate_world_size(count)
            if args.total_threads < count or args.total_threads % count:
                raise ValueError("--total-threads must be a positive multiple of each process count.")
        if args.dim < 2 or min(args.sizes) < args.dim:
            raise ValueError("Use dim >= 2 and at least dim rows per input.")
        if args.repeats < 3 or args.warmups < 0 or args.seed < 0:
            raise ValueError("Use at least three repeats and nonnegative warmups and seed.")
    except ValueError as error:
        parser.error(str(error))
    if args.worker:
        measure_worker(args)
        return

    with TemporaryDirectory(prefix="tsqr-benchmark-") as temp:
        measurements = []
        for rows in args.sizes:
            for algorithm, processes in [("qr", 1), *(("tsqr", p) for p in args.processes)]:
                path = Path(temp) / f"{rows}-{algorithm}-{processes}.json"
                measurements.append(run_config(args, rows, processes, algorithm, path))
    write_report(
        {
            "device": "cpu",
            "dtype": "float64",
            "platform": platform.system(),
            "torch_version": str(torch.__version__),
            "dim": args.dim,
            "total_threads": args.total_threads,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seed": args.seed,
            "measurements": measurements,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
