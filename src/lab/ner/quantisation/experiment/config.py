"""Validated configuration for controlled compression experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentIdentity(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    seed: int = 42


class DatasetConfig(BaseModel):
    path: Path
    split: str = "test"
    reference: Path | None = None


class InferenceConfig(BaseModel):
    batch_size: int = Field(default=16, ge=1)
    max_length: int = Field(default=256, ge=2)
    device: str = "auto"
    warmup_batches: int = Field(default=10, ge=0)
    repetitions: int = Field(default=5, ge=1)
    pad_to_multiple_of: int | None = Field(default=8, ge=1)
    # The canonical inference helper currently fixes Trainer workers at zero.
    # Reject rather than silently ignore a different experimental condition.
    number_of_workers: Literal[0] = 0


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    path: Path
    runtime: str = "pytorch"
    backend: str | None = None
    execution_provider: str | None = None
    representation: str = "huggingface"
    batch_size: int | None = Field(default=None, ge=1)


class EnergyConfig(BaseModel):
    enabled: bool = False
    modes: list[Literal["end_to_end", "model_only"]] = Field(
        default_factory=lambda: ["end_to_end"]
    )
    min_duration_seconds: float = Field(default=60.0, ge=0)
    measure_power_secs: int = Field(default=1, ge=1)


class OutputConfig(BaseModel):
    directory: Path


class StatisticsConfig(BaseModel):
    bootstrap_iterations: int = Field(default=10_000, ge=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    seed: int = 42


class AnalysisConfig(BaseModel):
    training_frequency_source: Path | None = None
    seen_definition: Literal["normalized_mention_string"] = "normalized_mention_string"
    plots: bool = False


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment: ExperimentIdentity
    dataset: DatasetConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    models: list[ModelConfig] = Field(min_length=1)
    energy: EnergyConfig = Field(default_factory=EnergyConfig)
    output: OutputConfig
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    protocol: Literal["controlled", "maximum_throughput"] = "controlled"

    @model_validator(mode="after")
    def unique_models(self) -> "ExperimentConfig":
        identifiers = [model.id for model in self.models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Model ids must be unique.")
        if self.protocol == "controlled" and not any(model.id == "A0" for model in self.models):
            raise ValueError("A controlled experiment requires the A0 baseline.")
        if self.protocol == "controlled" and any(model.batch_size is not None for model in self.models):
            raise ValueError("Per-model batch sizes are forbidden in a controlled experiment.")
        if self.protocol == "controlled" and len({model.runtime for model in self.models}) != 1:
            raise ValueError("All models must use the same runtime in a controlled experiment.")
        providers = {model.execution_provider for model in self.models}
        if self.protocol == "controlled" and len(providers) != 1:
            raise ValueError(
                "All models must use the same execution provider in a controlled experiment."
            )
        return self


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate one experiment YAML file."""
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return ExperimentConfig.model_validate(data)


def write_resolved(config: ExperimentConfig, path: str | Path) -> Path:
    """Write a fully resolved, portable YAML snapshot."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return destination
