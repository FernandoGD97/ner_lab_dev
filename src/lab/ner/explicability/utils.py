"""Small filesystem, distributed, and serialization utilities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def stable_id(kind: str, *parts: object) -> str:
    """Return a deterministic, compact ID with unambiguous component encoding."""
    payload = json.dumps([kind, *parts], ensure_ascii=False, separators=(",", ":"))
    return f"{kind}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def distributed_context() -> tuple[int, int]:
    """Rank/world size from initialized torch.distributed, then launcher variables."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except (ImportError, RuntimeError):
        pass
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    return rank, max(world, 1)


def distributed_barrier() -> None:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except (ImportError, RuntimeError):
        return


def directory_size(path: str | Path) -> int:
    root = Path(path)
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file()) if root.exists() else 0


def available_bytes(path: str | Path) -> int:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(target).free


def atomic_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path
