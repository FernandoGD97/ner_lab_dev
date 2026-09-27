"""Configuration scaffold; expensive distillation is intentionally not launched in Phase 1."""
from dataclasses import dataclass
@dataclass(frozen=True)
class DistillationWeights:
 hard_labels:float=1.; logits:float=1.; hidden_states:float=0.; attentions:float=0.; intermediate:float=0.
class DistillationTrainer:
 def __init__(self,*args,**kwargs): self.args=args; self.kwargs=kwargs
 def train(self): raise NotImplementedError('Distillation execution belongs to Phase 2; configure students/losses here.')
