"""Small framework-neutral NER distillation training loop."""
from __future__ import annotations
from dataclasses import dataclass
from .losses import attention_loss, hidden_state_loss, logits_kd, supervised_loss

@dataclass(frozen=True)
class DistillationWeights:
    hard_labels: float = 1.; logits: float = 1.; hidden_states: float = 0.; attentions: float = 0.
    def validate(self):
        values=(self.hard_labels,self.logits,self.hidden_states,self.attentions)
        if any(value < 0 for value in values) or not any(values): raise ValueError('Distillation weights must be non-negative and at least one must be active.')

class DistillationTrainer:
    """Train a student from batches of model keyword tensors.

    The teacher is permanently frozen and evaluated under ``no_grad``.  This
    deliberately accepts an iterable/DataLoader so the canonical NER encoder is
    responsible for corpus preprocessing rather than duplicating it here.
    """
    def __init__(self,teacher,student,optimizer,weights=DistillationWeights(),temperature=2.,layer_mapping=(),projections=None,ignore_index=-100):
        self.teacher=teacher.eval().requires_grad_(False);self.student=student;self.optimizer=optimizer
        self.weights=weights;weights.validate();self.temperature=temperature;self.layer_mapping=tuple(layer_mapping)
        self.projections=projections or {};self.ignore_index=ignore_index
    def step(self,batch):
        import torch
        labels=batch.get('labels'); mask=batch.get('attention_mask')
        request_hidden=self.weights.hidden_states>0;request_attention=self.weights.attentions>0
        inputs={key:value for key,value in batch.items() if key!='labels'}
        with torch.no_grad():
            teacher=self.teacher(**inputs,output_hidden_states=request_hidden,output_attentions=request_attention)
        student=self.student(**inputs,output_hidden_states=request_hidden,output_attentions=request_attention)
        loss=student.logits.sum()*0
        if self.weights.hard_labels:
            if labels is None: raise ValueError('Hard-label loss requires labels.')
            loss=loss+self.weights.hard_labels*supervised_loss(student.logits,labels,self.ignore_index)
        token_mask=(labels!=self.ignore_index) if labels is not None else mask
        if self.weights.logits: loss=loss+self.weights.logits*logits_kd(student.logits,teacher.logits,self.temperature,token_mask)
        if self.weights.hidden_states:
            if not self.layer_mapping: raise ValueError('Hidden-state loss requires an explicit layer mapping.')
            terms=[hidden_state_loss(student.hidden_states[s+1],teacher.hidden_states[t+1],mask,
                    self.projections.get(s)) for s,t in enumerate(self.layer_mapping)]
            loss=loss+self.weights.hidden_states*torch.stack(terms).mean()
        if self.weights.attentions:
            if not self.layer_mapping: raise ValueError('Attention loss requires an explicit layer mapping.')
            terms=[attention_loss(student.attentions[s],teacher.attentions[t],mask) for s,t in enumerate(self.layer_mapping)]
            loss=loss+self.weights.attentions*torch.stack(terms).mean()
        self.optimizer.zero_grad();loss.backward();self.optimizer.step();return float(loss.detach())
    def train(self,batches,epochs=1):
        if epochs<1:raise ValueError('epochs must be positive.')
        self.student.train();return [self.step(batch) for _ in range(epochs) for batch in batches]
