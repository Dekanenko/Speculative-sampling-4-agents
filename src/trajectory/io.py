"""JSONL read/write for trajectories.

File format (one JSON object per line):
    line 0      metadata header with {"__kind__": "metadata", ...}
    line 1..N   trajectory steps with {"__kind__": "step", ...}

The ``__kind__`` discriminator lets the reader dispatch each line to
the right dataclass without peeking at the index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Trajectory, TrajectoryMetadata, TrajectoryStep


_META_KIND = "metadata"
_STEP_KIND = "step"


def write_trajectory(path: str | Path, trajectory: Trajectory) -> None:
    """Write a trajectory to a JSONL file.

    Args:
        path: Destination file path. Parent directories must exist.
        trajectory: The trajectory to serialise.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        meta = trajectory.metadata.to_dict()
        meta["__kind__"] = _META_KIND
        fh.write(json.dumps(meta, ensure_ascii=False))
        fh.write("\n")
        for step in trajectory.steps:
            payload = step.to_dict()
            payload["__kind__"] = _STEP_KIND
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")


def read_trajectory(path: str | Path) -> Trajectory:
    """Read a trajectory from a JSONL file.

    Args:
        path: Path to a JSONL file previously written by ``write_trajectory``.

    Returns:
        A reconstructed ``Trajectory``.

    Raises:
        ValueError: If the file is empty or the metadata header is missing.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        lines = [line for line in fh.read().splitlines() if line]

    if not lines:
        raise ValueError(f"Trajectory file {path} is empty")

    header = json.loads(lines[0])
    if header.get("__kind__") != _META_KIND:
        raise ValueError(
            f"Expected metadata header as first line of {path}, got "
            f"{header.get('__kind__')!r}"
        )
    header.pop("__kind__")
    metadata = TrajectoryMetadata(**header)

    steps: list[TrajectoryStep] = []
    for raw in lines[1:]:
        payload: dict[str, Any] = json.loads(raw)
        if payload.get("__kind__") != _STEP_KIND:
            raise ValueError(
                f"Unexpected line kind {payload.get('__kind__')!r} in {path}"
            )
        payload.pop("__kind__")
        steps.append(TrajectoryStep(**payload))

    return Trajectory(metadata=metadata, steps=steps)
