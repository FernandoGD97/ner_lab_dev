"""Deterministic calibration-based reference W8A8 quantisation.

This backend stores weights as INT8 and applies calibrated activation fake
quantisation before a floating linear operation.  It is a portable correctness
backend, not an optimized kernel backend, and metadata states that distinction.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..artifacts import prepare_output, write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import CompressionCategory, Fidelity, MethodMetadata, Publication, TrainingRequirement
from ..model import load_checkpoint

STATIC_INT8_META = MethodMetadata(
    "int8-static", "quantization", CompressionCategory.MODEL_COMPRESSION,
    TrainingRequirement.CALIBRATION, fidelity=Fidelity.GENERIC_EQUIVALENT,
    limitations=("Reference W8A8 backend dequantises for Linear compute and does not claim optimized latency.",
                 "CPU only; calibration text must be explicitly supplied."),
    publications=(Publication("jacob-quant", "Quantization and Training of Neural Networks",
                              "https://arxiv.org/abs/1712.05877"),),
)


def read_calibration_texts(path: str | Path) -> list[str]:
    """Read a local parquet corpus, JSON/JSONL records, or UTF-8 text lines."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration corpus does not exist: {path}")
    if path.suffix == ".parquet":
        import pandas as pd
        frame = pd.read_parquet(path)
        if "text" not in frame:
            raise ValueError("Calibration parquet must contain a 'text' column.")
        values = frame["text"].dropna().astype(str).tolist()
    elif path.suffix in {".json", ".jsonl"}:
        raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(path.read_text())
        raw = raw if isinstance(raw, list) else raw.get("documents", [])
        values = [str(item["text"] if isinstance(item, dict) else item) for item in raw]
    else:
        values = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError("Calibration corpus contains no non-empty text.")
    return values


def _replace(model, ranges, prefix=""):
    import torch
    replaced = []
    for name, child in list(model.named_children()):
        qualified = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and qualified in ranges:
            scale = child.weight.detach().abs().amax().clamp_min(torch.finfo(torch.float32).eps) / 127
            qweight = torch.round(child.weight.detach() / scale).clamp(-127, 127).to(torch.int8)
            activation_scale = max(float(ranges[qualified]) / 127, torch.finfo(torch.float32).eps)

            class ReferenceStaticLinear(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.register_buffer("qweight", qweight.cpu())
                    self.register_buffer("weight_scale", scale.cpu()); self.activation_scale = activation_scale
                    self.bias = torch.nn.Parameter(child.bias.detach().cpu()) if child.bias is not None else None
                def forward(self, inputs):
                    qinput = torch.round(inputs / self.activation_scale).clamp(-127, 127)
                    inputs = qinput * self.activation_scale
                    weight = self.qweight.to(inputs.device, inputs.dtype) * self.weight_scale.to(inputs.device)
                    return torch.nn.functional.linear(inputs, weight, self.bias)
            setattr(model, name, ReferenceStaticLinear()); replaced.append(qualified)
        else:
            replaced.extend(_replace(child, ranges, qualified))
    return replaced


def calibrate(model, tokenizer, texts, batch_size=8, max_length=256):
    import torch
    ranges = {}; handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            def hook(_module, args, _output, key=name):
                value = float(args[0].detach().abs().amax())
                ranges[key] = max(ranges.get(key, 0.0), value)
            handles.append(module.register_forward_hook(hook))
    model.cpu().eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            model(**encoded)
    for handle in handles: handle.remove()
    return ranges


class StaticINT8Method(CompressionMethod):
    metadata = STATIC_INT8_META
    def check_compatibility(self, source, **kwargs):
        return [] if kwargs.get("calibration_corpus") else ["calibration_corpus is required for static INT8."]
    def apply(self, source, output, calibration_corpus, task="auto", calibration_samples=128,
              calibration_seed=42, batch_size=8, max_length=256, **kwargs):
        import torch
        if calibration_samples < 1 or batch_size < 1: raise ValueError("Calibration sample size and batch size must be positive.")
        texts = read_calibration_texts(calibration_corpus)
        indices = list(range(len(texts))); random.Random(calibration_seed).shuffle(indices)
        texts = [texts[index] for index in indices[:calibration_samples]]
        out = prepare_output(source, output); model, tokenizer, resolved = load_checkpoint(source, task)
        logical_parameters = sum(parameter.numel() for parameter in model.parameters())
        ranges = calibrate(model, tokenizer, texts, batch_size, max_length)
        replaced = _replace(model, ranges)
        if not replaced: raise ValueError("Calibration found no Linear modules to quantise.")
        torch.save(model.state_dict(), out / "static_int8_state.pt")
        model.config.save_pretrained(out); tokenizer.save_pretrained(out)
        write_artifact_metadata(source, out, self.metadata.name, self.metadata,
            backend="pytorch_reference_static_int8", parameter_count=logical_parameters,
            details={"task": resolved, "runtime_file": "static_int8_state.pt", "bit_width": 8,
                     "weights": "int8 symmetric per-tensor", "activations": "int8 symmetric calibrated per-module",
                     "quantized_modules": replaced, "activation_ranges": ranges,
                     "calibration": {"corpus": str(Path(calibration_corpus).resolve()), "samples": len(texts),
                                     "seed": calibration_seed, "batch_size": batch_size, "max_length": max_length},
                     "supported_devices": ["cpu"]})
        return out


def load_static_int8(path):
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
    path = Path(path); details = json.loads((path / "compression_manifest.json").read_text())["details"]
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    classes = {"base": AutoModel, "token-classification": AutoModelForTokenClassification,
               "sequence-classification": AutoModelForSequenceClassification}
    model = classes[details["task"]].from_config(config)
    _replace(model, details["activation_ranges"])
    model.load_state_dict(torch.load(path / details["runtime_file"], map_location="cpu", weights_only=True))
    return model
