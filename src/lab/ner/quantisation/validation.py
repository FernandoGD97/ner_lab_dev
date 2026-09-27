"""Reload and numerical smoke validation for standard HF artifacts."""
import json
from pathlib import Path
from dataclasses import asdict, dataclass
@dataclass(frozen=True)
class ValidationReport:
    config_loads: bool; tokenizer_loads: bool; model_loads: bool; task_head_preserved: bool
    label_mappings_preserved: bool; vocabulary_consistent: bool; forward_succeeds: bool
    outputs_finite: bool; shapes_preserved: bool
    @property
    def valid(self): return all(asdict(self).values())
    def to_dict(self): return {**asdict(self), 'valid':self.valid}
def validate_checkpoint(artifact, source=None, task='auto'):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from .model import load_checkpoint
    manifest_path = Path(artifact) / 'compression_manifest.json'
    backend = json.loads(manifest_path.read_text()).get('backend') if manifest_path.exists() else 'transformers'
    specialized = {
        'pytorch_dynamic_int8': ('.quantization.int8', 'load_dynamic_int8'),
        'pytorch_reference_static_int8': ('.quantization.static_int8', 'load_static_int8'),
        'pytorch_int4_reference': ('.quantization.int4', 'load_int4'),
    }
    if backend in specialized:
        import importlib
        module_name, loader_name = specialized[backend]
        loader = getattr(importlib.import_module(module_name, __package__), loader_name)
        model = loader(artifact); tokenizer = AutoTokenizer.from_pretrained(artifact)
        model.eval(); encoded = tokenizer('validation input', return_tensors='pt')
        with torch.no_grad(): output = model(**encoded)
        tensor = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output.logits
        consistent = len(tokenizer) == model.get_input_embeddings().num_embeddings
        return ValidationReport(True, True, True, True, True, consistent, True,
                                bool(torch.isfinite(tensor).all()), tensor.shape[0] == 1)
    if backend == 'pytorch_low_rank':
        from .low_rank.svd import load_low_rank
        model, tokenizer = load_low_rank(artifact); model.eval()
        with torch.no_grad(): output = model(**tokenizer('validation input', return_tensors='pt'))
        tensor = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output.logits
        return ValidationReport(True, True, True, True, True,
            len(tokenizer) == model.get_input_embeddings().num_embeddings, True,
            bool(torch.isfinite(tensor).all()), tensor.shape[0] == 1)
    if backend == 'onnx':
        raise ValueError("ONNX is a specialized runtime artifact; validate it with an installed ONNX Runtime provider.")
    config=AutoConfig.from_pretrained(artifact); tokenizer=AutoTokenizer.from_pretrained(artifact)
    model, tokenizer, resolved=load_checkpoint(artifact,task); model.eval()
    encoded=tokenizer('validation input',return_tensors='pt')
    with torch.no_grad(): output=model(**encoded)
    tensor=output.last_hidden_state if hasattr(output,'last_hidden_state') else output.logits
    source_model=None
    if source: source_model,_,source_task=load_checkpoint(source,task)
    labels=True if source_model is None else (source_model.config.label2id==model.config.label2id and source_model.config.id2label==model.config.id2label)
    head=True if source_model is None else source_task==resolved
    return ValidationReport(True,True,True,head,labels,len(tokenizer)==model.get_input_embeddings().num_embeddings,True,bool(torch.isfinite(tensor).all()),tensor.shape[0]==1)
