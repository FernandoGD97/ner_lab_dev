"""Fresh-process execution of one model × repetition benchmark run."""
from __future__ import annotations

import gc
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lab.core.provenance import write_manifest
from lab.core.segmentation import split_into_sentences
from lab.core.spans import spans_from_corpus
from lab.ner.encoding.encoder import encoder_from_description
from lab.ner.evaluation.scoring import span_metrics
from lab.ner.inference import (
    decode_spans,
    has_gold,
    load_model,
    predict_logits,
    read_documents,
    read_reference,
    write_gold,
    write_predictions,
)

from ..model import checkpoint_size
from .energy import EnergyTracker
from .schemas import MemoryMetrics, RunStatus, TimingMetrics, failure_status
from .tokenizer import tokenizer_statistics


def synchronize(device) -> None:
    """Synchronize the selected CUDA device before/after wall-clock timing."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(function: Callable[[], Any], device) -> tuple[Any, float]:
    synchronize(device)
    started = time.perf_counter()
    value = function()
    synchronize(device)
    return value, time.perf_counter() - started


def current_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / 1024 if sys.platform != "darwin" else usage / 1024**2


def count_padded_tokens(rows, batch_size: int, pad_to_multiple_of: int | None) -> int:
    total = 0
    lengths = [len(ids) for ids in rows["input_ids"]]
    for start in range(0, len(lengths), batch_size):
        batch = lengths[start:start + batch_size]
        padded = max(batch)
        if pad_to_multiple_of:
            padded = math.ceil(padded / pad_to_multiple_of) * pad_to_multiple_of
        total += padded * len(batch)
    return total


def collate_model_batches(rows, tokenizer, batch_size, pad_to_multiple_of):
    """Collate fixed host inputs once for pure model-forward measurement."""
    from transformers import DataCollatorForTokenClassification
    from lab.ner.encoding.rows import IGNORE_INDEX

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )
    columns = [
        column for column in ("input_ids", "attention_mask", "token_type_ids", "labels")
        if column in rows.columns
    ]
    batches = []
    for start in range(0, len(rows), batch_size):
        records = rows.iloc[start:start + batch_size][columns].to_dict("records")
        batch = collator(records)
        batches.append(dict(batch))
    return batches


def move_batches(batches, device):
    return [{key: value.to(device) for key, value in batch.items()} for batch in batches]


def forward_batches(model, batches) -> None:
    import torch

    with torch.inference_mode():
        for batch in batches:
            model(**batch)


def memory_metrics(device) -> dict[str, Any]:
    import torch

    values = MemoryMetrics(cpu_rss_mb=current_rss_mb())
    result = values.__dict__.copy()
    if device.type == "cuda":
        result.update({
            "gpu_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "gpu_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        })
    return result


def model_metadata(model, tokenizer, artifact: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    parameters = list(model.parameters())
    config = model.config
    observed_parameters = sum(parameter.numel() for parameter in parameters)
    parameter_count = manifest.get("parameter_count", observed_parameters)
    nonzero_parameters = None
    if manifest.get("backend", "transformers") == "transformers":
        nonzero_parameters = sum(int(parameter.count_nonzero()) for parameter in parameters)
    return {
        "parameters": parameter_count,
        "nonzero_parameters": nonzero_parameters,
        "checkpoint_bytes": checkpoint_size(artifact),
        "vocab_size": getattr(config, "vocab_size", len(tokenizer)),
        "hidden_size": getattr(config, "hidden_size", getattr(config, "dim", None)),
        "intermediate_size": getattr(config, "intermediate_size", getattr(config, "hidden_dim", None)),
        "layers": getattr(config, "num_hidden_layers", getattr(config, "n_layers", None)),
        "attention_heads": getattr(config, "num_attention_heads", getattr(config, "n_heads", None)),
        "precision": manifest.get("dtype", str(next(iter(parameters)).dtype).removeprefix("torch.")),
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    run_dir = Path(payload["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    run_file = run_dir / "run.json"
    running = {**payload["identity"], "status": RunStatus.RUNNING.value}
    write_manifest(running, run_file)

    model = tokenizer = None
    try:
        model_path = Path(payload["model_path"])
        manifest_path = model_path / "compression_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        backend = payload.get("backend") or manifest.get("backend", "transformers")
        device = torch.device(
            "cuda" if payload["device"] == "auto" and torch.cuda.is_available()
            else "cpu" if payload["device"] == "auto"
            else payload["device"]
        )
        if backend == "pytorch_dynamic_int8" and device.type != "cpu":
            raise ValueError("Dynamic INT8 is CPU-only and cannot use the controlled CUDA device.")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        (model, tokenizer, description), load_time = timed(
            lambda: load_model(model_path, device=payload["device"]), device
        )
        encoding = description["encoding"]
        configured_max = int(payload["max_length"])
        artifact_max = int(encoding.get("max_length", configured_max))
        if artifact_max != configured_max:
            raise ValueError(
                f"Controlled max_length={configured_max} differs from artifact encoding max_length={artifact_max}."
            )

        corpus, preprocessing_time = timed(
            lambda: read_documents(payload["dataset"]), device
        )
        encoder = encoder_from_description(
            encoding, tokenizer, require_target_label=False
        )
        rows, tokenization_time = timed(lambda: encoder.encode(corpus), device)
        if rows.empty:
            raise ValueError("The fixed dataset produced no model input windows.")

        batch_size = int(payload["batch_size"])
        padding = payload["pad_to_multiple_of"]
        host_batches = collate_model_batches(rows, tokenizer, batch_size, padding)
        batches, h2d_time = timed(
            lambda: move_batches(host_batches, device),
            device,
        )
        for index in range(int(payload["warmup_batches"])):
            forward_batches(model, [batches[index % len(batches)]])
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        cuda_event_seconds = None
        start_event = end_event = None
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        _, model_time = timed(lambda: forward_batches(model, batches), device)
        if start_event is not None and end_event is not None:
            end_event.record()
            torch.cuda.synchronize(device)
            cuda_event_seconds = start_event.elapsed_time(end_event) / 1000

        predictions, pipeline_inference_time = timed(
            lambda: predict_logits(
                model, tokenizer, rows, batch_size=batch_size,
                device=payload["device"], pad_to_multiple_of=padding,
            ),
            device,
        )

        texts = dict(zip(corpus["doc_id"], corpus["text"]))
        spans, postprocessing_time = timed(
            lambda: decode_spans(
                rows, predictions, tokenizer, encoder.id2label, texts=texts
            ),
            device,
        )
        predictions_dir = Path(payload["predictions_dir"])
        predictions_dir.mkdir(parents=True, exist_ok=True)
        write_predictions(spans, predictions_dir / "predictions.tsv")

        gold = None
        if payload.get("reference"):
            gold = read_reference(payload["reference"], encoder.target_label)
        elif has_gold(corpus):
            gold = spans_from_corpus(corpus, label=encoder.target_label)
        metrics: dict[str, Any] = {}
        if gold is not None:
            write_gold(gold, predictions_dir / "gold.tsv")
            metrics = span_metrics(gold, spans, tags=[encoder.target_label])
        write_manifest(metrics, Path(payload["model_output_dir"]) / "metrics.json")

        n_documents = int(corpus["doc_id"].nunique())
        n_sentences = sum(
            len(split_into_sentences(text, encoding["language"])) for text in corpus["text"]
        )
        n_tokens = sum(len(ids) for ids in rows["input_ids"])
        n_padded_tokens = count_padded_tokens(rows, batch_size, padding)
        end_to_end = (
            preprocessing_time + tokenization_time + pipeline_inference_time + postprocessing_time
        )
        timings = TimingMetrics(
            load_time_s=load_time,
            preprocessing_time_s=preprocessing_time,
            tokenization_time_s=tokenization_time,
            h2d_time_s=h2d_time if device.type == "cuda" else None,
            model_time_s=model_time,
            pipeline_inference_time_s=pipeline_inference_time,
            cuda_event_model_time_s=cuda_event_seconds,
            postprocessing_time_s=postprocessing_time,
            end_to_end_time_s=end_to_end,
            cold_start_time_s=load_time + end_to_end,
            warmup_batches=payload["warmup_batches"],
        ).__dict__
        memory = memory_metrics(device)

        energy: dict[str, Any] = {}
        energy_error = None
        if payload["energy_enabled"]:
            for mode in payload["energy_modes"]:
                try:
                    tracker = EnergyTracker(mode, payload["measure_power_secs"])
                    tracker.start()
                    energy_started = time.perf_counter()
                    passes = 0
                    while passes == 0 or time.perf_counter() - energy_started < payload["min_duration_seconds"]:
                        if mode == "model_only":
                            forward_batches(model, batches)
                        else:
                            energy_corpus = read_documents(payload["dataset"])
                            energy_rows = encoder.encode(energy_corpus)
                            energy_predictions = predict_logits(model, tokenizer, energy_rows, batch_size=batch_size, device=payload["device"], pad_to_multiple_of=padding)
                            decode_spans(energy_rows, energy_predictions, tokenizer, encoder.id2label, texts=texts)
                        passes += 1
                    synchronize(device)
                    energy[mode] = tracker.stop(passes, passes * n_documents, passes * n_tokens)
                except Exception:
                    energy_error = traceback.format_exc()
                    energy[mode] = {}

        metadata = model_metadata(model, tokenizer, model_path, manifest)
        tokenizer_stats = tokenizer_statistics(tokenizer, corpus)
        precision = metrics.get("span_strict_precision")
        recall = metrics.get("span_strict_recall")
        f1 = metrics.get("span_strict_f1")
        primary_energy = energy.get("end_to_end") or energy.get("model_only") or {}
        primary_energy_mode = (
            "end_to_end" if energy.get("end_to_end") else
            "model_only" if energy.get("model_only") else None
        )
        row = {
            **payload["identity"],
            "artifact_id": payload["model_id"],
            "source_model": manifest.get("source_model", str(model_path)),
            "method_chain": json.dumps(manifest.get("method_chain", ["fp32"])),
            "model_representation": payload["representation"],
            "runtime": payload["runtime"], "backend": backend,
            "execution_provider": payload.get("execution_provider"),
            "protocol": payload["protocol"], "task": "ner",
            "dataset": payload["dataset"], "split": payload["split"],
            "batch_size": batch_size, "max_length": configured_max,
            "n_documents": n_documents, "n_sentences": n_sentences,
            "n_tokens": n_tokens, "n_padded_tokens": n_padded_tokens,
            **metadata,
            "precision_score": precision, "recall_score": recall, "f1_score": f1,
            **timings,
            "documents_per_s": n_documents / end_to_end if end_to_end else None,
            "sentences_per_s": n_sentences / end_to_end if end_to_end else None,
            "tokens_per_s": n_tokens / end_to_end if end_to_end else None,
            **memory, **primary_energy,
            "energy_mode": primary_energy_mode,
            "status": RunStatus.SUCCESS.value,
            "energy_modes": energy, "energy_error": energy_error,
            "tokenizer_statistics": tokenizer_stats,
        }
        write_manifest(timings, run_dir / "timing.json")
        write_manifest(memory, run_dir / "memory.json")
        write_manifest(energy, run_dir / "energy.json")
        if energy:
            import pandas as pd

            pd.DataFrame(
                [{"mode": mode, **values} for mode, values in energy.items()]
            ).to_csv(run_dir / "energy.csv", index=False)
        write_manifest(tokenizer_stats, Path(payload["model_output_dir"]) / "tokenizer_statistics.json")
        write_manifest({**manifest, **metadata}, Path(payload["model_output_dir"]) / "model_metadata.json")
        write_manifest(row, run_file)
        return row
    except Exception as error:
        row = {
            **payload["identity"],
            "status": failure_status(error).value,
            "traceback": traceback.format_exc(),
        }
        write_manifest(row, run_file)
        return row
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def main() -> int:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    row = run(payload)
    return 0 if row["status"] == RunStatus.SUCCESS.value else 1


if __name__ == "__main__":
    raise SystemExit(main())
