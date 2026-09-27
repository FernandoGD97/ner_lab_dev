"""Executable token-classification student construction."""
from ..artifacts import prepare_output, write_artifact_metadata
from ..base import CompressionMethod
from ..distillation.students import StudentSpec, build_student
from ..metadata import (
    CompressionCategory, Fidelity, MethodMetadata, Publication, TrainingRequirement,
)
from ..model import load_checkpoint

STUDENT_META = MethodMetadata(
    "student", "architecture", CompressionCategory.MODEL_COMPRESSION,
    TrainingRequirement.DISTILLATION, fidelity=Fidelity.GENERIC_EQUIVALENT,
    limitations=(
        "Constructs and initializes a student; quality recovery requires training or distillation.",
        "Currently supports AutoModelForTokenClassification-compatible encoder configurations.",
    ),
    publications=(Publication("distilbert", "DistilBERT", "https://arxiv.org/abs/1910.01108"),),
)


class StudentMethod(CompressionMethod):
    metadata = STUDENT_META

    def apply(self, source, output, layers, hidden_size, intermediate_size=None,
              attention_heads=None, task="token-classification", seed=42, **kwargs):
        import torch

        if task not in ("auto", "token-classification"):
            raise ValueError("Student construction currently supports token classification only.")
        spec = StudentSpec(layers, hidden_size, intermediate_size, attention_heads)
        spec.validate()
        torch.manual_seed(seed)
        out = prepare_output(source, output)
        teacher, tokenizer, resolved = load_checkpoint(source, "token-classification")
        student, report = build_student(teacher, spec)
        student.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        report["seed"] = seed
        report["task"] = resolved
        write_artifact_metadata(
            source, out, "student", self.metadata,
            details={"task": resolved, "student": report, "seed": seed},
            parameter_count=sum(parameter.numel() for parameter in student.parameters()),
        )
        self.validate(out)
        return out
