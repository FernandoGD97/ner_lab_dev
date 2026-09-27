"""Validated reproducible compression recipes."""
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict

class MethodConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    mode: str | None = None
class OutputConfig(BaseModel): path: Path
class CompressionRecipe(BaseModel):
    model: str
    method: MethodConfig
    output: OutputConfig
    status: str = "READY"

def load_recipe(path):
    with Path(path).open(encoding='utf8') as stream: data = yaml.safe_load(stream)
    return CompressionRecipe.model_validate(data)
