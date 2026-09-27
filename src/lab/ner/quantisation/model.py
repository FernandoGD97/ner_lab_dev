"""Checkpoint loading and architecture-neutral inspection."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ModelInspection:
    source: str; architecture: str; model_class: str; task: str
    parameter_count: int; trainable_parameters: int; dtype: str; checkpoint_bytes: int
    vocabulary_size: int | None; hidden_size: int | None; intermediate_size: int | None
    number_of_layers: int | None; attention_heads: int | None; number_of_labels: int | None
    tokenizer_class: str; special_tokens: dict[str, Any]
    def to_dict(self): return asdict(self)

def checkpoint_size(path: str | Path) -> int:
    p = Path(path)
    return sum(x.stat().st_size for x in p.rglob('*') if x.is_file()) if p.exists() else 0

def config_value(config, *names):
    for name in names:
        value = getattr(config, name, None)
        if value is not None: return value
    return None

def load_checkpoint(source: str | Path, task: str = "auto"):
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
    source = str(source); config = AutoConfig.from_pretrained(source)
    classes = {"base": AutoModel, "token-classification": AutoModelForTokenClassification,
               "sequence-classification": AutoModelForSequenceClassification}
    if task == "auto":
        architectures = " ".join(config.architectures or []).lower()
        task = "token-classification" if "tokenclassification" in architectures else ("sequence-classification" if "sequenceclassification" in architectures else "base")
    model = classes[task].from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    return model, tokenizer, task

def inspect_checkpoint(source: str | Path, task: str = "auto") -> ModelInspection:
    model, tokenizer, task = load_checkpoint(source, task)
    config = model.config; parameters = list(model.parameters())
    dtype_names = sorted({str(p.dtype).removeprefix('torch.') for p in parameters})
    return ModelInspection(str(source), getattr(config, 'model_type', type(model).__name__), type(model).__name__, task,
        sum(p.numel() for p in parameters), sum(p.numel() for p in parameters if p.requires_grad),
        ','.join(dtype_names), checkpoint_size(source), config_value(config, 'vocab_size'),
        config_value(config, 'hidden_size', 'dim'), config_value(config, 'intermediate_size', 'hidden_dim'),
        config_value(config, 'num_hidden_layers', 'n_layers'), config_value(config, 'num_attention_heads', 'n_heads'),
        config_value(config, 'num_labels'), type(tokenizer).__name__, dict(tokenizer.special_tokens_map))
