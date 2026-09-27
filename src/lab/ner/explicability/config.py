"""Validated configuration for longitudinal representation snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


CORE_ANALYSES = (
    "temporal_projection", "trajectories", "centroid_dynamics", "cka", "geometry",
    "intrinsic_dimension", "neighborhoods", "training_dynamics", "forgetting",
    "transitions", "probing", "parameter_drift", "representation_shifts",
    "performance_relationships", "hard_examples", "layer_specialization",
)
OPTIONAL_ANALYSES = (
    "svcca", "pwcca", "mdl_probing", "attribution", "influence", "topology",
    "adaptation_phases",
)


@dataclass(frozen=True)
class AnalysisConfig:
    temporal_projection: bool = True
    trajectories: bool = True
    centroid_dynamics: bool = True
    cka: bool = True
    geometry: bool = True
    intrinsic_dimension: bool = True
    neighborhoods: bool = True
    training_dynamics: bool = True
    forgetting: bool = True
    transitions: bool = True
    probing: bool = True
    parameter_drift: bool = True
    representation_shifts: bool = True
    performance_relationships: bool = True
    hard_examples: bool = True
    layer_specialization: bool = True
    svcca: bool = False
    pwcca: bool = False
    mdl_probing: bool = False
    attribution: bool = False
    influence: bool = False
    topology: bool = False
    adaptation_phases: bool = False
    projection_method: str = "shared_pca"
    projection_dimensions: int = 2
    neighbors_k: int = 10
    probe_seed: int = 42
    probe_train_fraction: float = 0.7
    probe_validation_fraction: float = 0.15
    probe_max_iterations: int = 200
    probe_learning_rate: float = 0.1
    minimum_group_size: int = 3
    block_size: int = 1024
    workers: int = 1
    sample_size: int | None = None
    sample_seed: int = 42

    def __post_init__(self) -> None:
        if self.projection_method not in {"shared_pca", "fixed_umap", "aligned_umap"}:
            raise ValueError("projection_method must be shared_pca, fixed_umap, or aligned_umap.")
        if self.projection_dimensions < 1 or self.neighbors_k < 1 or self.block_size < 1:
            raise ValueError("Projection dimensions, neighbors_k, and block_size must be positive.")
        if not 0 < self.probe_train_fraction < 1:
            raise ValueError("probe_train_fraction must lie between zero and one.")
        if not 0 <= self.probe_validation_fraction < 1:
            raise ValueError("probe_validation_fraction must lie in [0, 1).")
        if self.probe_train_fraction + self.probe_validation_fraction >= 1:
            raise ValueError("Probe train and validation fractions must leave a test split.")
        if self.minimum_group_size < 2 or self.workers < 1:
            raise ValueError("minimum_group_size must be >=2 and workers must be positive.")
        if self.sample_size is not None and self.sample_size < 2:
            raise ValueError("sample_size must be at least two when configured.")

    def enabled(self) -> tuple[str, ...]:
        return tuple(name for name in (*CORE_ANALYSES, *OPTIONAL_ANALYSES) if getattr(self, name))


@dataclass(frozen=True)
class VisualizationConfig:
    profile: str = "paper"
    static_format: str = "svg"
    pdf: bool = False
    png: bool = False
    palette: str = "okabe_ito"
    background: str = "white"
    grid: bool = False
    top_spine: bool = False
    right_spine: bool = False
    smart_legend: bool = True
    outside_legend_when_dense: bool = True
    dense_scatter_rasterized: bool = True
    raster_dpi: int = 600
    visual_sample_size: int = 20_000
    visual_sample_seed: int = 42
    animation: bool = True
    animation_fps: int = 2
    animation_gif: bool = False
    optional_formats: Mapping[str, bool] | None = None
    spines: Mapping[str, bool] | None = None
    legend: Mapping[str, bool] | None = None
    rasterization: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.profile not in {"paper", "presentation"}:
            raise ValueError("visualization.profile must be paper or presentation.")
        if self.static_format != "svg":
            raise ValueError("SVG is the mandatory canonical static format.")
        if self.palette not in {"okabe_ito", "tol_bright", "colorbrewer_dark2"}:
            raise ValueError("Unknown color-blind-safe visualization palette.")
        if self.background.lower() != "white" or self.grid:
            raise ValueError("Scientific defaults require a white background and grid=false.")
        if self.top_spine or self.right_spine:
            raise ValueError("Scientific defaults require hidden top and right spines.")
        if self.raster_dpi < 72 or self.visual_sample_size < 100 or self.animation_fps < 1:
            raise ValueError("Invalid raster DPI, visual sample size, or animation FPS.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "VisualizationConfig":
        raw = dict(value or {})
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown visualization settings: {sorted(unknown)}")
        optional = dict(raw.get("optional_formats") or {})
        spines = dict(raw.get("spines") or {})
        legend = dict(raw.get("legend") or {})
        raster = dict(raw.get("rasterization") or {})
        nested_unknown = {
            "optional_formats": set(optional) - {"pdf", "png"},
            "spines": set(spines) - {"top", "right"},
            "legend": set(legend) - {"smart_position", "outside_when_dense"},
            "rasterization": set(raster) - {"dense_scatter", "dpi"},
        }
        invalid = {name: sorted(keys) for name, keys in nested_unknown.items() if keys}
        if invalid:
            raise ValueError(f"Unknown nested visualization settings: {invalid}")
        translated = {
            **raw,
            "pdf": optional.get("pdf", raw.get("pdf", False)),
            "png": optional.get("png", raw.get("png", False)),
            "top_spine": spines.get("top", raw.get("top_spine", False)),
            "right_spine": spines.get("right", raw.get("right_spine", False)),
            "smart_legend": legend.get("smart_position", raw.get("smart_legend", True)),
            "outside_legend_when_dense": legend.get(
                "outside_when_dense", raw.get("outside_legend_when_dense", True)
            ),
            "dense_scatter_rasterized": raster.get(
                "dense_scatter", raw.get("dense_scatter_rasterized", True)
            ),
            "raster_dpi": raster.get("dpi", raw.get("raster_dpi", 600)),
        }
        return cls(**{key: value for key, value in translated.items()
                      if key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SnapshotConfig:
    trigger: str = "evaluation"
    split: str = "validation"
    embedding_scope: str = "last_hidden_state"
    selected_layers: tuple[int, ...] = ()
    dtype: str = "float16"
    include_all_examples: bool = True
    include_all_tokens: bool = True
    cleanup_policy: str = "on_success"
    batch_size: int | None = None
    chunk_size: int = 16
    disk_safety_margin_bytes: int = 2 * 1024**3
    metadata_overhead_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.trigger not in {"evaluation", "epoch"}:
            raise ValueError("explicability.snapshots.trigger must be 'evaluation' or 'epoch'.")
        if self.split != "validation":
            raise ValueError("Part 1 supports snapshots.split='validation' only.")
        if self.embedding_scope not in {
            "last_hidden_state", "selected_layers", "all_hidden_states"
        }:
            raise ValueError("Unknown explicability embedding_scope.")
        if self.embedding_scope == "selected_layers" and not self.selected_layers:
            raise ValueError("selected_layers cannot be empty when embedding_scope='selected_layers'.")
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("explicability snapshot dtype must be float16 or float32.")
        if self.cleanup_policy not in {"on_success", "never"}:
            raise ValueError("cleanup_policy must be 'on_success' or 'never'.")
        if not self.include_all_examples:
            raise ValueError("Part 1 does not support scientific example sampling.")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if self.disk_safety_margin_bytes < 0 or self.metadata_overhead_fraction < 0:
            raise ValueError("Disk safety settings cannot be negative.")


@dataclass(frozen=True)
class ExplicabilityConfig:
    enabled: bool = False
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)
    analyses: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_config(value: ExplicabilityConfig | Mapping[str, Any] | None) -> ExplicabilityConfig:
    """Resolve the YAML-shaped mapping accepted by public training functions."""
    if value is None:
        return ExplicabilityConfig()
    if isinstance(value, ExplicabilityConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("explicability must be a mapping or ExplicabilityConfig.")

    unknown = set(value) - {"enabled", "snapshots", "analyses", "visualization"}
    if unknown:
        raise ValueError(f"Unknown explicability settings: {sorted(unknown)}")
    raw_snapshots = value.get("snapshots", {})
    if not isinstance(raw_snapshots, Mapping):
        raise TypeError("explicability.snapshots must be a mapping.")
    allowed = set(SnapshotConfig.__dataclass_fields__)
    extra = set(raw_snapshots) - allowed
    if extra:
        raise ValueError(f"Unknown explicability snapshot settings: {sorted(extra)}")
    settings = dict(raw_snapshots)
    settings["selected_layers"] = tuple(settings.get("selected_layers", ()))
    raw_analyses = value.get("analyses", {})
    if not isinstance(raw_analyses, Mapping):
        raise TypeError("explicability.analyses must be a mapping.")
    analysis_allowed = set(AnalysisConfig.__dataclass_fields__)
    analysis_extra = set(raw_analyses) - analysis_allowed
    if analysis_extra:
        raise ValueError(f"Unknown explicability analysis settings: {sorted(analysis_extra)}")
    return ExplicabilityConfig(
        enabled=bool(value.get("enabled", False)),
        snapshots=SnapshotConfig(**settings),
        analyses=AnalysisConfig(**dict(raw_analyses)),
        visualization=VisualizationConfig.from_mapping(value.get("visualization")),
    )
