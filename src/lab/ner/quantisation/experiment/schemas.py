"""Stable schemas and statuses for individual benchmark runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ModelReadiness(str, Enum):
    READY = "READY"
    MISSING = "MISSING"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    OOM = "OOM"
    SKIPPED = "SKIPPED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class TimingMetrics:
    load_time_s: float | None = None
    preprocessing_time_s: float | None = None
    tokenization_time_s: float | None = None
    h2d_time_s: float | None = None
    model_time_s: float | None = None
    pipeline_inference_time_s: float | None = None
    cuda_event_model_time_s: float | None = None
    postprocessing_time_s: float | None = None
    end_to_end_time_s: float | None = None
    cold_start_time_s: float | None = None
    warmup_batches: int = 0


@dataclass(frozen=True)
class MemoryMetrics:
    cpu_rss_mb: float | None = None
    gpu_allocated_mb: float | None = None
    gpu_reserved_mb: float | None = None
    gpu_peak_allocated_mb: float | None = None
    gpu_peak_reserved_mb: float | None = None


@dataclass(frozen=True)
class EnergyMetrics:
    mode: str
    cpu_energy_kwh: float | None = None
    gpu_energy_kwh: float | None = None
    ram_energy_kwh: float | None = None
    total_energy_kwh: float | None = None
    duration_s: float | None = None
    co2_kg: float | None = None
    passes: int = 0
    documents_processed: int = 0
    tokens_processed: int = 0
    joules_per_1000_documents: float | None = None
    joules_per_1000_tokens: float | None = None
    wh_per_1000_documents: float | None = None


def serializable(value: Any) -> dict[str, Any]:
    return asdict(value)


def failure_status(error: BaseException) -> RunStatus:
    """Classify OOM without changing batch size or swallowing other failures."""
    name = type(error).__name__.lower()
    message = str(error).lower()
    return RunStatus.OOM if "outofmemory" in name or "out of memory" in message else RunStatus.FAILED
