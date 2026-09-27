"""Pydantic EL extension reusing Phase 2 identity, energy, and output schemas."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field, model_validator
from ..experiment.config import EnergyConfig, ExperimentIdentity, OutputConfig

class ELTaskConfig(BaseModel):
    type: Literal["entity_linking"] = "entity_linking"
    mentions: Path
    gazetteer: Path
    training_mentions: Path | None = None

class RetrievalConfig(BaseModel):
    candidate_k: list[int] = Field(default_factory=lambda:[200,100,50,25,10])
    method: str = "matrix"
    index_type: Literal["flat_ip","flat_l2","scalar_quantization","product_quantization","ivf","ivf_pq","hnsw"] = "flat_ip"
    distance_metric: Literal["ip","l2","cosine"] = "cosine"
    method_kwargs: dict = Field(default_factory=dict)
    @model_validator(mode="after")
    def valid_k(self):
        if not self.candidate_k or any(k<1 for k in self.candidate_k): raise ValueError("candidate_k values must be positive.")
        if len(self.candidate_k)!=len(set(self.candidate_k)): raise ValueError("candidate_k values must be unique.")
        if self.index_type == "flat_l2" and self.distance_metric != "l2":
            raise ValueError("flat_l2 requires distance_metric='l2'.")
        if self.index_type == "flat_ip" and self.distance_metric == "l2":
            raise ValueError("flat_ip requires distance_metric='ip' or 'cosine'.")
        return self

class BiEncoderConfig(BaseModel):
    id: str = "biencoder"
    model: str | None = None
    embedding_dimension: int | None = Field(default=None,ge=1)

class CrossEncoderConfig(BaseModel):
    id: str = "crossencoder"
    enabled: bool = False
    model: Path | None = None
    batch_size: int = Field(default=16,ge=1)
    @model_validator(mode="after")
    def model_required(self):
        if self.enabled and self.model is None: raise ValueError("Enabled crossencoder requires model.")
        return self

class ELAnalysisConfig(BaseModel):
    seen_unseen: bool = True
    frequency_buckets: bool = True

class ELExperimentConfig(BaseModel):
    experiment: ExperimentIdentity
    task: ELTaskConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    biencoder: BiEncoderConfig = Field(default_factory=BiEncoderConfig)
    crossencoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)
    analysis: ELAnalysisConfig = Field(default_factory=ELAnalysisConfig)
    energy: EnergyConfig = Field(default_factory=EnergyConfig)
    output: OutputConfig
    repetitions: int = Field(default=5,ge=1)

def load_el_experiment(path):
    with Path(path).open(encoding="utf-8") as stream:return ELExperimentConfig.model_validate(yaml.safe_load(stream))
