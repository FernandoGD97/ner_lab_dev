"""Longitudinal Transformer representation capture for NER."""

from lab.ner.explicability.config import (
    AnalysisConfig, ExplicabilityConfig, SnapshotConfig, VisualizationConfig,
    resolve_config,
)
from lab.ner.explicability.finalize import finalize_run, initialize_run, validate_manifest
from lab.ner.explicability.pooling import pool_groups, pool_spans
from lab.ner.explicability.snapshot import capture_snapshot, estimate_snapshot_bytes
from lab.ner.explicability.storage import iter_chunks, read_table, write_table


def run_analyses(*args, **kwargs):
    """Lazily import the Part 2 pipeline so Part 1 utilities remain lightweight."""
    from lab.ner.explicability.analyses import run_analyses as run
    return run(*args, **kwargs)


def generate_figures(*args, **kwargs):
    """Lazily import the SVG/report layer."""
    from lab.ner.explicability.visualization import generate_figures as generate
    return generate(*args, **kwargs)

__all__ = [
    "AnalysisConfig", "ExplicabilityConfig", "SnapshotConfig", "VisualizationConfig",
    "capture_snapshot",
    "estimate_snapshot_bytes",
    "finalize_run", "initialize_run", "iter_chunks", "pool_groups", "pool_spans",
    "generate_figures", "read_table", "resolve_config", "run_analyses",
    "validate_manifest", "write_table",
]
