"""Research-grade Transformer compression tools for :mod:`lab.ner`."""
from .base import CompressionMethod
from .model import ModelInspection, inspect_checkpoint
from .registry import get_method, list_methods
__all__=['CompressionMethod','ModelInspection','inspect_checkpoint','get_method','list_methods']
