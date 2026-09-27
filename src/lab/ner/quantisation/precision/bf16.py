from ..artifacts import prepare_output, write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import *
from ..model import load_checkpoint
BF16_META=MethodMetadata('bf16','precision',CompressionCategory.COMPUTATIONAL_COMPRESSION,TrainingRequirement.NONE,
 fidelity=Fidelity.EXACT,limitations=('Efficient execution requires BF16-capable hardware.',),
 publications=(Publication('bfloat16','A Study of BFLOAT16 for Deep Learning Training','https://arxiv.org/abs/1905.12322'),),reference_implementations=('https://pytorch.org/docs/stable/tensor_attributes.html',))
class BF16Method(CompressionMethod):
 metadata=BF16_META
 def apply(self,source,output,task='auto',**kwargs):
  import torch
  out=prepare_output(source,output); model,tok,_=load_checkpoint(source,task); model.to(dtype=torch.bfloat16); model.save_pretrained(out,safe_serialization=True); tok.save_pretrained(out)
  write_artifact_metadata(source,out,'bf16',self.metadata); self.validate(out); return out
