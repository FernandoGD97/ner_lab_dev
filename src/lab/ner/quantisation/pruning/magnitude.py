from ..artifacts import prepare_output,write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import *
from ..model import load_checkpoint
MAGNITUDE_META=MethodMetadata('magnitude-pruning','pruning',CompressionCategory.MODEL_COMPRESSION,TrainingRequirement.FINE_TUNING,
 fidelity=Fidelity.GENERIC_EQUIVALENT,limitations=('Unstructured zeros do not by themselves reduce dense checkpoint bytes, FLOPs, or latency.',),
 publications=(Publication('deep-compression','Deep Compression','https://arxiv.org/abs/1510.00149'),),reference_implementations=('https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html',))
class MagnitudePruningMethod(CompressionMethod):
 metadata=MAGNITUDE_META
 def apply(self,source,output,amount=.2,task='auto',**kwargs):
  import torch
  out=prepare_output(source,output); model,tok,_=load_checkpoint(source,task); pruned=total=0
  with torch.no_grad():
   for p in model.parameters():
    if p.ndim < 2: continue
    count=int(p.numel()*amount)
    if count: threshold=p.abs().flatten().kthvalue(count).values; mask=p.abs()<=threshold; pruned+=int(mask.sum()); total+=p.numel(); p.masked_fill_(mask,0)
  model.save_pretrained(out,safe_serialization=True); tok.save_pretrained(out)
  write_artifact_metadata(source,out,'magnitude-pruning',self.metadata,details={'requested_fraction':amount,'zeroed_parameters':pruned,'eligible_parameters':total}); self.validate(out); return out
