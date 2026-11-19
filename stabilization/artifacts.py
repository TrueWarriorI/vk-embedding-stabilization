"""Atomic writes for small experiment artifacts."""

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def artifact_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            yield temporary
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_report(report: dict, path: Path | None) -> None:
    output = json.dumps(report, indent=2, allow_nan=False)
    if path is not None:
        with artifact_writer(path) as file:
            file.write((output + "\n").encode("utf-8"))
    print(output, flush=True)
