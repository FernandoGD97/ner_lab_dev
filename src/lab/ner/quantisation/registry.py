"""Single registry of implemented and declared methods."""
from .metadata import *
from .precision import FP16Method,BF16Method
from .quantization.int8 import DynamicINT8Method
from .pruning import MagnitudePruningMethod,VocabularyPruningMethod
from .architecture import DepthReductionMethod
from .low_rank import SVDMethod
from .runtime import ONNXMethod
BASELINE_META=MethodMetadata('fp32','baseline',CompressionCategory.COMPUTATIONAL_COMPRESSION,TrainingRequirement.NONE,
 fidelity=Fidelity.EXACT,limitations=('Baseline only; no transformation is applied.',),publications=(Publication('bert','BERT','https://arxiv.org/abs/1810.04805'),))
_METHODS={m.metadata.name:m for m in (FP16Method(),BF16Method(),DynamicINT8Method(),MagnitudePruningMethod(),VocabularyPruningMethod(),DepthReductionMethod(),SVDMethod(),ONNXMethod())}
def list_methods(): return [BASELINE_META]+[x.metadata for x in _METHODS.values()]
def get_method(name):
 try:return _METHODS[name]
 except KeyError: raise ValueError(f"Unsupported compression method {name!r}. Available implemented methods: {', '.join(sorted(_METHODS))}.") from None
def method_metadata(name):
 for item in list_methods():
  if item.name==name:return item
 raise ValueError(f'Unknown compression method {name!r}.')
