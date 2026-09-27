"""Layer-prefix reduction for compatible encoder stacks (experimental, fine-tuning expected)."""
from ..artifacts import prepare_output,write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import *
from ..model import load_checkpoint
DEPTH_META=MethodMetadata('depth-reduction','architecture',CompressionCategory.MODEL_COMPRESSION,TrainingRequirement.FINE_TUNING,
 fidelity=Fidelity.PAPER_INSPIRED,limitations=('Keeps a prefix of layers; quality requires fine-tuning/distillation.','Only conventional encoder layer lists are supported.'),
 publications=(Publication('layerdrop','Reducing Transformer Depth on Demand with Structured Dropout','https://arxiv.org/abs/1909.11556'),))
def _layers(model):
 base=model.base_model
 for owner,name in ((getattr(base,'encoder',None),'layer'),(getattr(base,'transformer',None),'layer')):
  if owner is not None and hasattr(owner,name): return owner,name,getattr(owner,name)
 raise ValueError(f'Unsupported encoder stack for {type(model).__name__}.')
class DepthReductionMethod(CompressionMethod):
 metadata=DEPTH_META
 def apply(self,source,output,layers,task='auto',**kwargs):
  import torch
  out=prepare_output(source,output); model,tok,_=load_checkpoint(source,task); owner,name,stack=_layers(model)
  if not 0 < layers < len(stack): raise ValueError(f'layers must be between 1 and {len(stack)-1}.')
  setattr(owner,name,torch.nn.ModuleList(list(stack)[:layers]));
  if hasattr(model.config,'num_hidden_layers'): model.config.num_hidden_layers=layers
  if hasattr(model.config,'n_layers'): model.config.n_layers=layers
  model.save_pretrained(out,safe_serialization=True); tok.save_pretrained(out)
  write_artifact_metadata(source,out,'depth-reduction',self.metadata,details={'kept_layers':layers}); self.validate(out); return out
