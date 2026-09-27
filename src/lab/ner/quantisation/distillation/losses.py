"""Memory-conscious token, representation, and attention KD objectives."""
from __future__ import annotations


def logits_kd(student_logits, teacher_logits, temperature=2.0, mask=None):
    """Temperature-scaled token KL divergence with the standard ``T²`` correction."""
    import torch.nn.functional as F
    if temperature <= 0: raise ValueError("temperature must be positive.")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher and student logits must have identical shapes.")
    student = student_logits / temperature; teacher = teacher_logits.detach() / temperature
    per_class = F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="none")
    per_token = per_class.sum(dim=-1)
    if mask is not None:
        selected = per_token[mask.to(dtype=__import__('torch').bool, device=per_token.device)]
        if selected.numel() == 0: return per_token.sum() * 0
        per_token = selected
    return per_token.mean() * temperature ** 2


def hidden_state_loss(student, teacher, mask=None, projection=None):
    import torch
    if projection is not None: student = projection(student)
    teacher = teacher.detach()
    if student.shape != teacher.shape: raise ValueError("Hidden states require equal shapes or a projection.")
    error = (student - teacher).pow(2).mean(dim=-1)
    return error[mask.bool()].mean() if mask is not None else error.mean()


def attention_loss(student, teacher, mask=None):
    import torch
    teacher = teacher.detach()
    if student.shape != teacher.shape: raise ValueError("Attention maps are incompatible.")
    error = (student - teacher).pow(2)
    if mask is not None:
        pair_mask = mask.bool()[:, None, :, None] & mask.bool()[:, None, None, :]
        selected = error[pair_mask.expand_as(error)]
        return selected.mean() if selected.numel() else error.sum() * 0
    return error.mean()


def supervised_loss(logits, labels, ignore_index=-100):
    import torch.nn.functional as F
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index)
