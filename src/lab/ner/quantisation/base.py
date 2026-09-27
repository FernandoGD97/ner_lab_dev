"""Unified API used by both Python callers and the CLI."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from .validation import validate_checkpoint
from .errors import UnsupportedMethodError
class CompressionMethod(ABC):
    metadata = None
    def check_compatibility(self, source, **kwargs): return []
    @abstractmethod
    def apply(self, source, output, **kwargs): ...
    def validate(self, artifact): return validate_checkpoint(artifact)
    def __call__(self, source, output, **kwargs):
        errors=self.check_compatibility(source, **kwargs)
        if errors: raise UnsupportedMethodError('; '.join(errors))
        return self.apply(source, output, **kwargs)
