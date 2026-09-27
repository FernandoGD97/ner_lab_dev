"""Composable losses for future distillation training."""
def logits_kd(student_logits,teacher_logits,temperature=2.0):
 import torch.nn.functional as F
 return F.kl_div(F.log_softmax(student_logits/temperature,dim=-1),F.softmax(teacher_logits/temperature,dim=-1),reduction='batchmean')*(temperature**2)
def hidden_state_loss(student,teacher):
 import torch.nn.functional as F
 return F.mse_loss(student,teacher)
def attention_loss(student,teacher): return hidden_state_loss(student,teacher)
