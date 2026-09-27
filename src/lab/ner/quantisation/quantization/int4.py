"""Portable weight-only INT4 artifact and inference module."""
from __future__ import annotations

from pathlib import Path

from ..artifacts import prepare_output, write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import CompressionCategory, Fidelity, MethodMetadata, Publication, TrainingRequirement
from ..model import load_checkpoint
from .packing import PackedInt4, dequantize_int4, quantize_int4

INT4_META = MethodMetadata(
    "int4", "quantization", CompressionCategory.MODEL_COMPRESSION,
    TrainingRequirement.NONE, fidelity=Fidelity.GENERIC_EQUIVALENT,
    limitations=("Portable reference backend dequantises weights for each forward pass; it does not claim INT4 kernel speed-up.",
                 "CPU inference only."),
    publications=(Publication("gptq", "GPTQ", "https://arxiv.org/abs/2210.17323"),),
)


def _torch():
    import torch
    return torch


class WeightOnlyInt4Linear:  # constructed dynamically as nn.Module below
    """Marker used for documentation and type discovery."""


def make_int4_linear(linear, group_size: int, symmetric: bool):
    torch = _torch()

    class _Int4Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            packed = quantize_int4(linear.weight, group_size, symmetric)
            self.register_buffer("packed_weight", packed.packed.cpu())
            self.register_buffer("scales", packed.scales.cpu())
            if packed.zero_points is not None:
                self.register_buffer("zero_points", packed.zero_points.cpu())
            else:
                self.zero_points = None
            self.shape = packed.shape
            self.padded_values = packed.padded_values
            self.group_size = packed.group_size
            self.symmetric = packed.symmetric
            self.in_features = linear.in_features
            self.out_features = linear.out_features
            self.bias = torch.nn.Parameter(linear.bias.detach().cpu()) if linear.bias is not None else None

        def forward(self, inputs):
            value = PackedInt4(self.packed_weight, self.scales, self.zero_points, self.shape,
                               self.padded_values, self.group_size, self.symmetric)
            weight = dequantize_int4(value, device=inputs.device, dtype=inputs.dtype)
            return torch.nn.functional.linear(inputs, weight, self.bias)

    return _Int4Linear()


def _replace_linears(module, group_size, symmetric, include, exclude, prefix=""):
    torch = _torch(); replaced = []
    for name, child in list(module.named_children()):
        qualified = f"{prefix}.{name}" if prefix else name
        selected = isinstance(child, torch.nn.Linear)
        selected &= not include or any(pattern in qualified for pattern in include)
        selected &= not any(pattern in qualified for pattern in exclude)
        if selected:
            setattr(module, name, make_int4_linear(child, group_size, symmetric)); replaced.append(qualified)
        else:
            replaced.extend(_replace_linears(child, group_size, symmetric, include, exclude, qualified))
    return replaced


class WeightOnlyINT4Method(CompressionMethod):
    metadata = INT4_META

    def apply(self, source, output, task="auto", group_size=64, symmetric=True,
              include_modules=(), excluded_modules=("classifier",), **kwargs):
        torch = _torch()
        if group_size < 1:
            raise ValueError("group_size must be positive.")
        out = prepare_output(source, output)
        model, tokenizer, resolved = load_checkpoint(source, task)
        model.cpu().eval()
        logical_parameters = sum(parameter.numel() for parameter in model.parameters())
        replaced = _replace_linears(model, group_size, symmetric, tuple(include_modules),
                                    tuple(excluded_modules))
        if not replaced:
            raise ValueError("No Linear modules matched the INT4 selection rules.")
        # State dictionaries are stable across Python processes; serialising the
        # dynamically constructed module itself would not be.
        torch.save(model.state_dict(), out / "int4_state.pt")
        model.config.save_pretrained(out); tokenizer.save_pretrained(out)
        write_artifact_metadata(
            source, out, self.metadata.name, self.metadata, backend="pytorch_int4_reference",
            details={"task": resolved, "runtime_file": "int4_state.pt", "bit_width": 4,
                     "group_size": group_size, "symmetric": symmetric,
                     "granularity": "per-group", "quantized_modules": replaced,
                     "excluded_modules": list(excluded_modules), "supported_devices": ["cpu"]},
            parameter_count=logical_parameters,
        )
        return out


def load_int4(path):
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
    import json

    torch = _torch(); path = Path(path)
    manifest = json.loads((path / "compression_manifest.json").read_text())
    details = manifest["details"]
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    classes = {"base": AutoModel, "token-classification": AutoModelForTokenClassification,
               "sequence-classification": AutoModelForSequenceClassification}
    model = classes[details["task"]].from_config(config)
    names = set(details["quantized_modules"])
    _replace_linears(model, details["group_size"], details["symmetric"], tuple(names), ())
    model.load_state_dict(torch.load(path / details["runtime_file"], map_location="cpu", weights_only=True))
    return model
