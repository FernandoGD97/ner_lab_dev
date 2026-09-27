"""Raw-results-first Phase 3 analysis orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from lab.core.provenance import file_sha256

from .errors import compare_predictions
from .pareto import export_pareto
from .reporting import correlations, export_paper_tables, family_summary, generate_plots, model_level, paper_tables
from .robustness import FREQUENCY_DEFINITION, baseline_deltas, robustness_table, training_frequencies
from .statistics import paired_bootstrap


def read_spans(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def model_paths(results_dir: str | Path) -> dict[str, Path]:
    resolved = Path(results_dir) / "resolved_experiment.yaml"
    if not resolved.exists(): return {}
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return {str(item["id"]): Path(item["path"]) for item in data.get("models", [])}


def _tokenizer(path: Path | None):
    if path is None: return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        return None


def analyse_results(
    results_dir: str | Path,
    training_frequency_source: str | Path | None = None,
    baseline: str = "A0",
    bootstrap_iterations: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
    plots: bool = False,
) -> dict[str, Any]:
    root = Path(results_dir); analysis = root / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    resolved_path = root / "resolved_experiment.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) if resolved_path.exists() else {}
    configured_analysis = resolved.get("analysis", {})
    configured_statistics = resolved.get("statistics", {})
    if training_frequency_source is None and configured_analysis.get("training_frequency_source"):
        training_frequency_source = configured_analysis["training_frequency_source"]
    bootstrap_iterations = int(configured_statistics.get("bootstrap_iterations", bootstrap_iterations))
    confidence_level = float(configured_statistics.get("confidence_level", confidence_level))
    seed = int(configured_statistics.get("seed", seed))
    plots = bool(configured_analysis.get("plots", plots))
    long = pd.read_csv(root / "benchmark_long.tsv", sep="\t")
    model_ids = sorted(long.model_id.dropna().unique())
    paths = model_paths(root)
    frequencies = None
    if training_frequency_source:
        source = Path(training_frequency_source)
        training = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source, sep="\t")
        frequencies = training_frequencies(training)
    frequency_metadata = {
        "source": None if training_frequency_source is None else str(Path(training_frequency_source).resolve()),
        "source_sha256": None if training_frequency_source is None else file_sha256(training_frequency_source),
        "definition": FREQUENCY_DEFINITION,
        "test_set_used_for_frequency": False,
    }
    (analysis / "frequency_definition.json").write_text(json.dumps(frequency_metadata, indent=2)+"\n")

    gold_path = root / "models" / baseline / "predictions" / "gold.tsv"
    if not gold_path.exists(): raise FileNotFoundError(f"Baseline gold spans not found: {gold_path}")
    gold = read_spans(gold_path); robustness = {}; bootstraps = []
    baseline_predictions = read_spans(root / "models" / baseline / "predictions" / "predictions.tsv")
    for model_id in model_ids:
        prediction_path = root / "models" / model_id / "predictions" / "predictions.tsv"
        if not prediction_path.exists(): continue
        predicted = read_spans(prediction_path)
        table = robustness_table(gold, predicted, frequencies, _tokenizer(paths.get(model_id)))
        table.insert(0, "model_id", model_id); table.to_csv(analysis/f"robustness_{model_id}.tsv",sep="\t",index=False)
        robustness[model_id] = table.drop(columns="model_id")
        if model_id != baseline:
            result = paired_bootstrap(gold, baseline_predictions, predicted, bootstrap_iterations, confidence_level, seed)
            result["model_id"] = model_id; result["baseline_id"] = baseline; bootstraps.append(result)
    deltas = baseline_deltas(robustness, baseline); deltas.to_csv(analysis/"robustness.tsv",sep="\t",index=False)
    pd.DataFrame(bootstraps).to_csv(analysis/"paired_bootstrap.tsv",sep="\t",index=False)
    corr = correlations(long); corr.to_csv(analysis/"correlations.tsv",sep="\t",index=False)
    family = family_summary(long); family.to_csv(analysis/"compression_families.tsv",sep="\t",index=False)
    averaged = model_level(long); pareto_paths = export_pareto(averaged, analysis)
    paper = export_paper_tables(paper_tables(long, deltas, baseline), analysis)
    plot_paths = generate_plots(long, analysis) if plots else []
    return {"analysis_dir":str(analysis),"models":model_ids,"pareto":{k:str(v) for k,v in pareto_paths.items()},"paper":{k:str(v) for k,v in paper.items()},"plots":[str(path) for path in plot_paths]}


def compare_stored_predictions(
    results_dir: str | Path, baseline: str, model: str,
    training_frequency_source: str | Path | None = None,
) -> dict[str, Any]:
    root=Path(results_dir); output=root/"analysis"; output.mkdir(parents=True,exist_ok=True)
    prediction_root=root/"models"
    gold=read_spans(prediction_root/baseline/"predictions"/"gold.tsv")
    base=read_spans(prediction_root/baseline/"predictions"/"predictions.tsv")
    compressed=read_spans(prediction_root/model/"predictions"/"predictions.tsv")
    summary, detail=compare_predictions(gold,base,compressed)
    frequencies=None
    if training_frequency_source:
        source=Path(training_frequency_source); training=pd.read_parquet(source) if source.suffix==".parquet" else pd.read_csv(source,sep="\t"); frequencies=training_frequencies(training)
    from .robustness import normalize_mention
    detail["text"]=detail["text_baseline"].fillna(detail["text_compressed"])
    detail["baseline_prediction"]=detail["error_type_baseline"]
    detail["compressed_prediction"]=detail["error_type_compressed"]
    detail["entity_frequency"]=detail["text"].map(lambda text: None if frequencies is None else frequencies[normalize_mention(text)])
    detail["mention_length"]=detail["end"]-detail["start"]
    tokenizer=_tokenizer(model_paths(root).get(model))
    detail["subword_count"]=None if tokenizer is None else detail["text"].map(lambda text:len(tokenizer(str(text),add_special_tokens=False)["input_ids"]))
    detail=detail.rename(columns={"gold_label":"gold_label"})
    leading=["document_id","start","end","text","gold_label","baseline_prediction","compressed_prediction","difference_type","entity_frequency","mention_length","subword_count"]
    detail=detail[[*leading,*[column for column in detail if column not in leading]]]
    path=output/f"prediction_differences_{baseline}_vs_{model}.tsv"; detail.to_csv(path,sep="\t",index=False)
    summary.update({
        "training_frequency_source": None if training_frequency_source is None else str(Path(training_frequency_source).resolve()),
        "training_frequency_definition": FREQUENCY_DEFINITION if training_frequency_source else None,
        "test_set_used_for_frequency": False,
    })
    (output/f"prediction_differences_{baseline}_vs_{model}.json").write_text(json.dumps(summary,indent=2)+"\n")
    return {**summary,"details":str(path)}
