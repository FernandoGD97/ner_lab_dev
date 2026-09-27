"""Validated reproducible compression recipes."""
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict

class MethodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    mode: str | None = None
    task: str = "auto"
    amount: float | None = None
    layers: int | None = None
    rank_ratio: float | None = None
    keep_tokens: list[str] | None = None
    calibration_corpus: Path | None = None
    calibration_samples: int | None = None
    calibration_seed: int | None = None
    batch_size: int | None = None
    max_length: int | None = None
    group_size: int | None = None
    symmetric: bool | None = None
    include_modules: list[str] | None = None
    excluded_modules: list[str] | None = None
    head_amount: float | None = None
    ffn_amount: float | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    attention_heads: int | None = None
    seed: int | None = None
class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
class CompressionRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    method: MethodConfig
    output: OutputConfig
    status: str = "READY"

def load_recipe(path):
    with Path(path).open(encoding='utf8') as stream: data = yaml.safe_load(stream)
    return CompressionRecipe.model_validate(data)
