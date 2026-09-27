"""Materialize a provenance-bearing FP32 control artifact."""
from ..artifacts import prepare_output,write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import *
from ..model import load_checkpoint
FP32_META=MethodMetadata('fp32','baseline',CompressionCategory.COMPUTATIONAL_COMPRESSION,TrainingRequirement.NONE,
 fidelity=Fidelity.EXACT,limitations=('Control artifact only; no compression is performed.',),
 publications=(Publication('bert','BERT','https://arxiv.org/abs/1810.04805'),))
class FP32Method(CompressionMethod):
 metadata=FP32_META
 def apply(self,source,output,task='auto',**kwargs):
  out=prepare_output(source,output);model,tokenizer,resolved=load_checkpoint(source,task);model=model.float()
  model.save_pretrained(out,safe_serialization=True);tokenizer.save_pretrained(out)
  write_artifact_metadata(source,out,'fp32',self.metadata,details={'task':resolved,'representation':'fp32-control'},
   parameter_count=sum(p.numel() for p in model.parameters()))
  self.validate(out);return out
