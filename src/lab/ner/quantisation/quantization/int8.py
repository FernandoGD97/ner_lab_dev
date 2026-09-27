"""CPU dynamic INT8 PTQ; output uses a specialized PyTorch runtime."""
from pathlib import Path
from ..artifacts import prepare_output, write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import *
from ..model import load_checkpoint
INT8_META=MethodMetadata('int8-dynamic','quantization',CompressionCategory.COMPUTATIONAL_COMPRESSION,TrainingRequirement.NONE,
 fidelity=Fidelity.GENERIC_EQUIVALENT,limitations=('CPU-only dynamic quantization.','Artifact is not loadable with AutoModel; use load_dynamic_int8.'),
 publications=(Publication('jacob-quant','Quantization and Training of Neural Networks','https://arxiv.org/abs/1712.05877'),),
 reference_implementations=('https://pytorch.org/docs/stable/quantization.html',))
class DynamicINT8Method(CompressionMethod):
 metadata=INT8_META
 def apply(self,source,output,task='auto',**kwargs):
  import torch
  out=prepare_output(source,output); model,tok,resolved=load_checkpoint(source,task); model.cpu().eval()
  quantize=getattr(torch.ao.quantization,'quantize_dynamic',None) or torch.quantization.quantize_dynamic
  quantized=quantize(model,{torch.nn.Linear},dtype=torch.qint8)
  torch.save(quantized,out/'quantized_model.pt'); model.config.save_pretrained(out); tok.save_pretrained(out)
  write_artifact_metadata(source,out,'int8-dynamic',self.metadata,'pytorch_dynamic_int8',{'task':resolved,'runtime_file':'quantized_model.pt'})
  return out
def load_dynamic_int8(path):
 import torch
 return torch.load(Path(path)/'quantized_model.pt',map_location='cpu',weights_only=False)
