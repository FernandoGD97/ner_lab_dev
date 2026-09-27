from ..metadata import *
INT4_META=MethodMetadata('int4','quantization',CompressionCategory.COMPUTATIONAL_COMPRESSION,TrainingRequirement.CALIBRATION,
 fidelity=Fidelity.EXTERNAL_ADAPTER,limitations=('No safe Phase 1 encoder implementation; backend- and hardware-specific.',),
 publications=(Publication('gptq','GPTQ','https://arxiv.org/abs/2210.17323'),Publication('awq','AWQ','https://arxiv.org/abs/2306.00978')))
