"""Reproducibility metadata for benchmark studies."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(root: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_environment(runtime: str = "pytorch", execution_provider: str | None = None) -> dict[str, Any]:
    """Capture software, hardware, and relevant process settings without requiring CUDA."""
    memory_bytes = None
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError):
        pass

    gpu: dict[str, Any] = {
        "gpu_count": 0,
        "gpu_models": [],
        "gpu_driver": None,
        "cuda": None,
        "cudnn": None,
    }
    try:
        import torch

        gpu["cuda"] = torch.version.cuda
        gpu["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            gpu["gpu_count"] = torch.cuda.device_count()
            gpu["gpu_models"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
            try:
                gpu["gpu_driver"] = torch._C._cuda_getDriverVersion()
            except AttributeError:
                pass
    except ImportError:
        pass

    variables = {
        name: os.environ[name]
        for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        if name in os.environ
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu": platform.processor() or platform.machine(),
        "ram_bytes": memory_bytes,
        **gpu,
        "python": platform.python_version(),
        "pytorch": _version("torch"),
        "transformers": _version("transformers"),
        "tokenizers": _version("tokenizers"),
        "codecarbon": _version("codecarbon"),
        "ner_lab_git_commit": _git_commit(),
        "runtime": runtime,
        "execution_provider": execution_provider,
        "environment_variables": variables,
    }
