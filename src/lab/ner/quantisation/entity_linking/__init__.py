"""Controlled analysis adapters for the repository's existing NEL pipeline."""
from __future__ import annotations
import importlib
_EXPORTS={"run_el_experiment":"benchmark","summarize_el":"benchmark","validate_el":"benchmark","el_metrics":"metrics"}
__all__=sorted(_EXPORTS)
def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    return getattr(importlib.import_module(f"lab.ner.quantisation.entity_linking.{_EXPORTS[name]}"),name)
