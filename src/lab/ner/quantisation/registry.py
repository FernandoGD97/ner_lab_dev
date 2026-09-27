"""Single registry of implemented and declared methods."""
from .metadata import *
from .precision import FP32Method,FP16Method,BF16Method
from .quantization.int8 import DynamicINT8Method
from .quantization.int4 import WeightOnlyINT4Method
from .quantization.static_int8 import StaticINT8Method
from .pruning import MagnitudePruningMethod,VocabularyPruningMethod,StructuredPruningMethod
from .architecture import DepthReductionMethod,StudentMethod
from .low_rank import SVDMethod
from .runtime import ONNXMethod
_METHODS={m.metadata.name:m for m in (FP32Method(),FP16Method(),BF16Method(),DynamicINT8Method(),StaticINT8Method(),WeightOnlyINT4Method(),MagnitudePruningMethod(),StructuredPruningMethod(),VocabularyPruningMethod(),DepthReductionMethod(),StudentMethod(),SVDMethod(),ONNXMethod())}
def list_methods(): return [x.metadata for x in _METHODS.values()]
def get_method(name):
 try:return _METHODS[name]
 except KeyError: raise ValueError(f"Unsupported compression method {name!r}. Available implemented methods: {', '.join(sorted(_METHODS))}.") from None
def method_metadata(name):
 for item in list_methods():
  if item.name==name:return item
 raise ValueError(f'Unknown compression method {name!r}.')
