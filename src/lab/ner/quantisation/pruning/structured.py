"""Reproducible structural attention-head and FFN-channel pruning."""
from __future__ import annotations
from ..artifacts import prepare_output, write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import CompressionCategory, Fidelity, MethodMetadata, Publication, TrainingRequirement
from ..model import load_checkpoint

STRUCTURED_META = MethodMetadata('structured-pruning','pruning',CompressionCategory.MODEL_COMPRESSION,
 TrainingRequirement.FINE_TUNING,fidelity=Fidelity.GENERIC_EQUIVALENT,
 limitations=('Magnitude criterion only.','FFN pruning supports conventional BERT/RoBERTa encoder blocks.'),
 publications=(Publication('head-pruning','Are Sixteen Heads Really Better than One?','https://arxiv.org/abs/1905.10650'),))

def _encoder_layers(model):
 base=model.base_model
 for owner in (getattr(base,'encoder',None),getattr(base,'transformer',None)):
  if owner is not None and hasattr(owner,'layer'): return owner.layer
 raise ValueError(f'Unsupported encoder stack: {type(model).__name__}.')

def prune_attention_heads(model, amount: float):
 import torch
 if not 0 <= amount < 1: raise ValueError('head amount must be in [0, 1).')
 heads=getattr(model.config,'num_attention_heads',getattr(model.config,'n_heads',None))
 if heads is None or not hasattr(model,'prune_heads'): raise ValueError('Architecture has no structural head-pruning API.')
 count=int(heads*amount)
 if not count:return {}
 decisions={}
 for index,layer in enumerate(_encoder_layers(model)):
  attention=getattr(getattr(layer,'attention',None),'self',None); query=getattr(attention,'query',None)
  if query is None: raise ValueError('Head magnitude extraction is unsupported for this architecture.')
  size=query.out_features//heads
  scores=query.weight.detach().abs().reshape(heads,size,-1).sum(dim=(1,2))
  decisions[index]=torch.argsort(scores)[:count].tolist()
 model.prune_heads(decisions); return decisions

def prune_ffn_channels(model, amount: float):
 import torch
 if not 0 <= amount < 1: raise ValueError('FFN amount must be in [0, 1).')
 kept={}
 for index,layer in enumerate(_encoder_layers(model)):
  first=getattr(getattr(layer,'intermediate',None),'dense',None); second=getattr(getattr(layer,'output',None),'dense',None)
  if first is None or second is None: raise ValueError('FFN structural pruning supports BERT/RoBERTa-style layers only.')
  count=max(1,round(first.out_features*(1-amount)))
  indices=torch.argsort(first.weight.detach().abs().sum(dim=1),descending=True)[:count].sort().values
  new_first=torch.nn.Linear(first.in_features,count,bias=first.bias is not None)
  new_second=torch.nn.Linear(count,second.out_features,bias=second.bias is not None)
  with torch.no_grad():
   new_first.weight.copy_(first.weight[indices]); new_second.weight.copy_(second.weight[:,indices])
   if first.bias is not None:new_first.bias.copy_(first.bias[indices])
   if second.bias is not None:new_second.bias.copy_(second.bias)
  layer.intermediate.dense=new_first; layer.output.dense=new_second; kept[index]=indices.tolist()
 model.config.intermediate_size=count
 return kept

class StructuredPruningMethod(CompressionMethod):
 metadata=STRUCTURED_META
 def apply(self,source,output,task='auto',head_amount=0.,ffn_amount=0.,**kwargs):
  if head_amount==0 and ffn_amount==0:raise ValueError('At least one structural pruning amount must be non-zero.')
  if not 0 <= head_amount < 1 or not 0 <= ffn_amount < 1:raise ValueError('Structural pruning amounts must be in [0, 1).')
  out=prepare_output(source,output); model,tokenizer,resolved=load_checkpoint(source,task)
  original=sum(p.numel() for p in model.parameters())
  heads=prune_attention_heads(model,head_amount) if head_amount else {}
  channels=prune_ffn_channels(model,ffn_amount) if ffn_amount else {}
  result=sum(p.numel() for p in model.parameters())
  model.save_pretrained(out,safe_serialization=True);tokenizer.save_pretrained(out)
  write_artifact_metadata(source,out,self.metadata.name,self.metadata,
   details={'task':resolved,'head_amount':head_amount,'ffn_amount':ffn_amount,'removed_heads':heads,
            'kept_ffn_channels':channels,'original_parameter_count':original,'result_parameter_count':result},
   parameter_count=result)
  self.validate(out);return out
