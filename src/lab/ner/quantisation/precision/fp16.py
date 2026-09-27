from ..artifacts import prepare_output, write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import *
from ..model import load_checkpoint
FP16_META=MethodMetadata('fp16','precision',CompressionCategory.COMPUTATIONAL_COMPRESSION,TrainingRequirement.NONE,
 fidelity=Fidelity.EXACT,limitations=('CPU kernels may not support FP16 or improve latency.',),
 publications=(Publication('mixed-precision','Mixed Precision Training','https://arxiv.org/abs/1710.03740'),),reference_implementations=('https://pytorch.org/docs/stable/tensor_attributes.html',))
class FP16Method(CompressionMethod):
 metadata=FP16_META
 def apply(self,source,output,task='auto',**kwargs):
  out=prepare_output(source,output); model,tok,_=load_checkpoint(source,task); model.half(); model.save_pretrained(out,safe_serialization=True); tok.save_pretrained(out)
  write_artifact_metadata(source,out,'fp16',self.metadata); self.validate(out); return out
