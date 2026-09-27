"""Deterministic teacher/student layer alignment."""
from __future__ import annotations


def evenly_spaced_layer_mapping(teacher_layers: int, student_layers: int) -> tuple[int, ...]:
    if teacher_layers < 1 or student_layers < 1:
        raise ValueError("Teacher and student depth must be positive.")
    if student_layers > teacher_layers:
        raise ValueError("A reduced student cannot have more layers than its teacher.")
    if student_layers == 1: return (teacher_layers - 1,)
    return tuple(round(index * (teacher_layers - 1) / (student_layers - 1)) for index in range(student_layers))
