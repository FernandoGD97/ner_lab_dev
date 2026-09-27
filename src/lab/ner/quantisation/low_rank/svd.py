"""Experimental SVD factorisation with an explicit specialized runtime."""
import json
from pathlib import Path
from ..artifacts import prepare_output,write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import *
from ..model import load_checkpoint
SVD_META=MethodMetadata('svd','low-rank',CompressionCategory.MODEL_COMPRESSION,TrainingRequirement.FINE_TUNING,
 fidelity=Fidelity.APPROXIMATION,limitations=('Specialized PyTorch artifact, not AutoModel-loadable.','Latency can worsen without fused low-rank kernels.'),
 publications=(Publication('svd-compression','Exploiting Linear Structure Within Convolutional Networks','https://arxiv.org/abs/1404.0736'),))
def _replace(model,rank_ratio):
 import torch
 replaced=[]
 for parent_name,parent in list(model.named_modules()):
  for name,child in list(parent.named_children()):
   if not isinstance(child,torch.nn.Linear): continue
   rank=max(1,min(child.in_features,child.out_features,int(min(child.in_features,child.out_features)*rank_ratio)))
   if rank*(child.in_features+child.out_features)>=child.in_features*child.out_features: continue
   u,s,vh=torch.linalg.svd(child.weight.detach().float(),full_matrices=False)
   first=torch.nn.Linear(child.in_features,rank,bias=False); second=torch.nn.Linear(rank,child.out_features,bias=child.bias is not None)
   first.weight.data.copy_(vh[:rank].to(child.weight.dtype)); second.weight.data.copy_((u[:,:rank]*s[:rank]).to(child.weight.dtype))
   if child.bias is not None: second.bias.data.copy_(child.bias.data)
   setattr(parent,name,torch.nn.Sequential(first,second)); replaced.append(f'{parent_name}.{name}'.strip('.'))
 return replaced
class SVDMethod(CompressionMethod):
 metadata=SVD_META
 def apply(self,source,output,rank_ratio=.5,task='auto',**kwargs):
  import torch
  if not 0 < rank_ratio < 1: raise ValueError('rank_ratio must be between 0 and 1.')
  out=prepare_output(source,output); model,tok,resolved=load_checkpoint(source,task); replaced=_replace(model,rank_ratio)
  if not replaced: raise ValueError('No linear layer benefits from the requested rank.')
  torch.save(model.state_dict(),out/'low_rank_state.pt'); model.config.save_pretrained(out); tok.save_pretrained(out)
  (out/'low_rank.json').write_text(json.dumps({'source':str(source),'task':resolved,'rank_ratio':rank_ratio,'layers':replaced},indent=2))
  count=sum(parameter.numel() for parameter in model.parameters())
  write_artifact_metadata(source,out,'svd',self.metadata,'pytorch_low_rank',{'rank_ratio':rank_ratio,'factorized_layers':replaced},parameter_count=count); return out
def load_low_rank(path):
 import torch
 data=json.loads((Path(path)/'low_rank.json').read_text()); model,tokenizer,_=load_checkpoint(data['source'],data['task']); _replace(model,data['rank_ratio']); model.load_state_dict(torch.load(Path(path)/'low_rank_state.pt',map_location='cpu',weights_only=True)); return model,tokenizer
