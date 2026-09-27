"""CPU-only checks for Phase 2 schemas, energy, failures, and tokenizer metrics."""
from lab.ner.quantisation.experiment.energy import normalize_energy
from lab.ner.quantisation.experiment.schemas import (
    MemoryMetrics,
    RunStatus,
    TimingMetrics,
    failure_status,
)
from lab.ner.quantisation.experiment.tokenizer import tokenizer_statistics

assert TimingMetrics(warmup_batches=3).warmup_batches == 3
assert MemoryMetrics(cpu_rss_mb=12.5).gpu_peak_allocated_mb is None
normalized = normalize_energy(0.001, documents=100, tokens=1000)
assert normalized["joules_per_1000_documents"] == 36000
assert normalized["joules_per_1000_tokens"] == 3600
assert normalized["wh_per_1000_documents"] == 10
assert failure_status(RuntimeError("CUDA out of memory")) == RunStatus.OOM
assert failure_status(ValueError("bad artifact")) == RunStatus.FAILED


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class Corpus:
    def __getitem__(self, columns):
        return self

    def itertuples(self, index=False):
        from types import SimpleNamespace
        return iter([
            SimpleNamespace(text="one two three", entities_json='[{"start": 0, "end": 3}]'),
            SimpleNamespace(text="four five", entities_json="[]"),
        ])


stats = tokenizer_statistics(Tokenizer(), Corpus())
assert stats["mean_tokens_per_document"] == 2.5
assert stats["median_tokens_per_document"] == 2.5
assert stats["p95_tokens_per_document"] == 3
assert stats["mean_subwords_per_word"] == 1
assert stats["mean_subwords_per_entity"] == 1
