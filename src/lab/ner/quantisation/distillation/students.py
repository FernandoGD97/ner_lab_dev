from dataclasses import dataclass
@dataclass(frozen=True)
class StudentSpec:
 layers:int; hidden_size:int; intermediate_size:int|None=None; attention_heads:int|None=None
 def validate(self):
  if self.layers<1 or self.hidden_size<1: raise ValueError('Student depth and width must be positive.')
  if self.attention_heads and self.hidden_size%self.attention_heads: raise ValueError('hidden_size must be divisible by attention_heads.')
