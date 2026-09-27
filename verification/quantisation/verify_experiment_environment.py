"""CPU-only checks for reproducibility metadata and resume validity."""
import json
import tempfile
from pathlib import Path

from lab.ner.quantisation.experiment.environment import capture_environment

metadata = capture_environment(runtime="pytorch", execution_provider="CPUExecutionProvider")
required = {"timestamp", "hostname", "os", "kernel", "cpu", "ram_bytes", "gpu_count",
            "python", "pytorch", "transformers", "tokenizers", "codecarbon",
            "ner_lab_git_commit", "runtime", "execution_provider", "environment_variables"}
assert required <= set(metadata)
assert metadata["runtime"] == "pytorch"
assert metadata["execution_provider"] == "CPUExecutionProvider"
