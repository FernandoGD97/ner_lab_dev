"""Student specifications and safe Hugging Face student construction."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from .mapping import evenly_spaced_layer_mapping

@dataclass(frozen=True)
class StudentSpec:
    layers: int; hidden_size: int; intermediate_size: int | None = None; attention_heads: int | None = None
    def validate(self):
        if self.layers < 1 or self.hidden_size < 1: raise ValueError("Student depth and width must be positive.")
        if self.intermediate_size is not None and self.intermediate_size < 1: raise ValueError("intermediate_size must be positive.")
        if self.attention_heads and self.hidden_size % self.attention_heads: raise ValueError("hidden_size must be divisible by attention_heads.")


def build_student(teacher, spec: StudentSpec):
    """Build a valid token-classification student and copy only shape-compatible tensors."""
    from copy import deepcopy
    from transformers import AutoModelForTokenClassification
    spec.validate(); config = deepcopy(teacher.config)
    teacher_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layers", None))
    teacher_hidden = getattr(config, "hidden_size", getattr(config, "dim", None))
    if teacher_layers is None or teacher_hidden is None: raise ValueError("Unsupported student configuration architecture.")
    resolved_heads = spec.attention_heads or getattr(config, "num_attention_heads", getattr(config, "n_heads", None))
    if resolved_heads is None or spec.hidden_size % resolved_heads:
        raise ValueError("Student hidden_size must be divisible by the configured attention-head count.")
    mapping = evenly_spaced_layer_mapping(teacher_layers, spec.layers)
    for name in ("num_hidden_layers", "n_layers"):
        if hasattr(config, name): setattr(config, name, spec.layers)
    for name in ("hidden_size", "dim"):
        if hasattr(config, name): setattr(config, name, spec.hidden_size)
    if spec.intermediate_size is not None:
        for name in ("intermediate_size", "hidden_dim"):
            if hasattr(config, name): setattr(config, name, spec.intermediate_size)
    if spec.attention_heads is not None:
        for name in ("num_attention_heads", "n_heads"):
            if hasattr(config, name): setattr(config, name, spec.attention_heads)
    student = AutoModelForTokenClassification.from_config(config)
    teacher_state = teacher.state_dict(); student_state = student.state_dict(); copied=[]; skipped=[]
    layer_markers = ("encoder.layer.", "transformer.layer.")
    for name, value in student_state.items():
        teacher_name = name
        for marker in layer_markers:
            if marker in name:
                prefix, suffix = name.split(marker, 1); index, remainder = suffix.split(".", 1)
                if index.isdigit() and int(index) < len(mapping):
                    teacher_name = f"{prefix}{marker}{mapping[int(index)]}.{remainder}"
                break
        source = teacher_state.get(teacher_name)
        if source is not None and source.shape == value.shape:
            value.copy_(source); copied.append(name)
        else: skipped.append(name)
    student.load_state_dict(student_state)
    return student, {"spec": asdict(spec), "layer_mapping": list(mapping), "copied_tensors": copied,
                     "randomly_initialized_tensors": skipped, "projected_tensors": [], "partially_copied_tensors": []}
