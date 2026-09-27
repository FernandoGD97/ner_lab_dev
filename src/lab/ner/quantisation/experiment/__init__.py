"""Controlled benchmark orchestration for compression experiments.

The public objects are loaded lazily so result-analysis utilities remain usable
without importing the heavy NER runtime.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "ExperimentConfig": "lab.ner.quantisation.experiment.config",
    "load_experiment": "lab.ner.quantisation.experiment.config",
    "validate_experiment": "lab.ner.quantisation.experiment.runner",
    "run_experiment": "lab.ner.quantisation.experiment.runner",
    "experiment_status": "lab.ner.quantisation.experiment.runner",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(importlib.import_module(_EXPORTS[name]), name)
