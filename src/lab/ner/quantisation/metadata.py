"""Scientific metadata shared by every compression implementation."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum

class CompressionCategory(str, Enum):
    MODEL_COMPRESSION = "model_compression"
    COMPUTATIONAL_COMPRESSION = "computational_compression"
    RUNTIME_OPTIMIZATION = "runtime_optimization"

class Fidelity(str, Enum):
    EXACT = "EXACT"
    APPROXIMATION = "APPROXIMATION"
    PAPER_INSPIRED = "PAPER_INSPIRED"
    GENERIC_EQUIVALENT = "GENERIC_EQUIVALENT"
    EXTERNAL_ADAPTER = "EXTERNAL_ADAPTER"

class TrainingRequirement(str, Enum):
    NONE = "none"
    CALIBRATION = "calibration"
    FINE_TUNING = "fine_tuning"
    DISTILLATION = "distillation"

@dataclass(frozen=True)
class Publication:
    key: str
    title: str
    url: str

@dataclass(frozen=True)
class MethodMetadata:
    name: str
    family: str
    category: CompressionCategory
    training_requirement: TrainingRequirement
    supported_architectures: tuple[str, ...] = ("BERT", "RoBERTa", "XLM-R", "DistilBERT")
    supported_tasks: tuple[str, ...] = ("base", "token-classification", "sequence-classification")
    fidelity: Fidelity = Fidelity.GENERIC_EQUIVALENT
    limitations: tuple[str, ...] = ()
    publications: tuple[Publication, ...] = ()
    reference_implementations: tuple[str, ...] = ()
    def to_dict(self):
        return asdict(self)
