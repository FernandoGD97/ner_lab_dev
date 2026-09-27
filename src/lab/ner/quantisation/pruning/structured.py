"""Structured pruning is reserved for a train-and-fine-tune phase."""
from ..metadata import *
STRUCTURED_META=MethodMetadata('structured-pruning','pruning',CompressionCategory.MODEL_COMPRESSION,TrainingRequirement.FINE_TUNING,fidelity=Fidelity.PAPER_INSPIRED,limitations=('Metadata/scaffold only in Phase 1.',),publications=(Publication('block-pruning','Block Movement Pruning','https://arxiv.org/abs/2102.06929'),))
