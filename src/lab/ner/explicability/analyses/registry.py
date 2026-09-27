"""Small explicit registry for independently selectable Part 2 analyses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    output: str
    expensive: bool
    function: Callable[..., Any] | None = None


REGISTRY: dict[str, AnalysisSpec] = {}

_DEFAULT_OUTPUTS = {
    "temporal_projection": "projection.parquet", "trajectories": "trajectories.parquet",
    "centroid_dynamics": "centroid_dynamics.parquet", "cka": "cka.parquet",
    "geometry": "geometry.parquet", "intrinsic_dimension": "intrinsic_dimension.parquet",
    "neighborhoods": "neighborhoods.parquet", "training_dynamics": "training_dynamics.parquet",
    "forgetting": "forgetting.parquet", "transitions": "transitions.parquet",
    "probing": "probing.parquet", "parameter_drift": "parameter_drift.parquet",
    "representation_shifts": "representation_shifts.parquet",
    "performance_relationships": "performance_relationships.parquet",
    "hard_examples": "hard_examples.parquet", "layer_specialization": "layer_specialization.parquet",
    "svcca": "svcca.parquet", "pwcca": "pwcca.parquet", "mdl_probing": "mdl_probing.parquet",
    "attribution": "attribution.parquet", "influence": "influence.parquet",
    "topology": "topology.parquet", "adaptation_phases": "adaptation_phases.parquet",
}
for _name, _output in _DEFAULT_OUTPUTS.items():
    REGISTRY[_name] = AnalysisSpec(
        _name, _output,
        _name in {"svcca", "pwcca", "mdl_probing", "attribution", "influence", "topology"},
    )


def register(name: str, output: str, *, expensive: bool = False):
    def decorator(function: Callable[..., Any]):
        if name in REGISTRY and REGISTRY[name].function is not None:
            raise ValueError(f"Analysis already registered: {name}")
        REGISTRY[name] = AnalysisSpec(name, output, expensive, function)
        return function
    return decorator


def analysis_registry() -> dict[str, AnalysisSpec]:
    return dict(REGISTRY)
