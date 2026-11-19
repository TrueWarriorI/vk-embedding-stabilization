"""Launch a single-node experiment through torchrun."""

import argparse
import os
import subprocess
import sys

from .runtime import validate_world_size


def torchrun_command(processes: int) -> tuple[list[str], dict]:
    validate_world_size(processes)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={processes}",
    ]
    env = os.environ.copy()
    if os.name == "nt":
        # Windows wheels may be built without the libuv TCPStore backend.
        env["USE_LIBUV"] = "0"
        env["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
        env["PET_SIGNALS_TO_HANDLE"] = "SIGINT,SIGTERM"
        command += ["--rdzv-backend=c10d", "--rdzv-conf=store_type=file", "--local-addr=127.0.0.1"]
    else:
        command.append("--standalone")
    command += ["--module", "stabilization"]
    return command, env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("experiment", nargs=argparse.REMAINDER, help="mode and experiment options")
    args = parser.parse_args()
    try:
        command, env = torchrun_command(args.processes)
    except ValueError as error:
        parser.error(str(error))
    return subprocess.run(command + args.experiment, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
