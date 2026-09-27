"""ONNX export is a runtime artifact, never represented as a normal HF checkpoint."""
from ..artifacts import prepare_output,write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import *
from ..model import load_checkpoint
ONNX_META=MethodMetadata('onnx','runtime',CompressionCategory.RUNTIME_OPTIMIZATION,TrainingRequirement.NONE,
 fidelity=Fidelity.EXTERNAL_ADAPTER,limitations=('Requires optional onnx package.','Runtime performance depends on ONNX Runtime/provider optimization.'),
 publications=(Publication('onnx','ONNX: Open Neural Network Exchange','https://onnx.ai/'),),reference_implementations=('https://huggingface.co/docs/transformers/serialization',))
class ONNXMethod(CompressionMethod):
 metadata=ONNX_META
 def apply(self,source,output,task='auto',opset=17,**kwargs):
  import importlib.util,torch
  if importlib.util.find_spec('onnx') is None: raise ValueError("ONNX export requires the optional 'onnx' package.")
  out=prepare_output(source,output); model,tok,resolved=load_checkpoint(source,task); model.cpu().eval(); inputs=tok('ONNX export',return_tensors='pt')
  names=list(inputs); torch.onnx.export(model,tuple(inputs[n] for n in names),out/'model.onnx',input_names=names,output_names=['output'],dynamic_axes={n:{0:'batch',1:'sequence'} for n in names},opset_version=opset)
  model.config.save_pretrained(out); tok.save_pretrained(out); write_artifact_metadata(source,out,'onnx',self.metadata,'onnx',{'task':resolved,'opset':opset}); return out
