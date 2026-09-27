"""Correlations, compression-family summaries, paper tables, and optional plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd

CORRELATION_PAIRS = [
    ("parameters", "checkpoint_bytes"), ("parameters", "model_time_s"),
    ("parameters", "total_energy_kwh"), ("checkpoint_bytes", "gpu_peak_allocated_mb"),
    ("checkpoint_bytes", "model_time_s"), ("estimated_flops", "model_time_s"),
    ("model_time_s", "total_energy_kwh"), ("gpu_peak_allocated_mb", "total_energy_kwh"),
]
FAMILY_KEYWORDS = {
    "precision": ("fp16", "bf16"), "quantization": ("int8", "int4", "quant"),
    "vocabulary pruning": ("vocabulary",), "structural pruning": ("structured",),
    "distillation": ("distill",), "architectural reduction": ("depth", "width", "ffn", "student"),
    "low rank": ("svd", "low-rank"), "runtime optimization": ("onnx", "runtime"),
}


def model_level(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns
    return frame[frame.status == "SUCCESS"].groupby("model_id", as_index=False)[list(numeric)].mean()


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    data = model_level(frame); output = []
    for left, right in CORRELATION_PAIRS:
        if left not in data or right not in data: continue
        pair = data[[left, right]].dropna()
        for method in ("pearson", "spearman"):
            if len(pair) < 2:
                value = None
            elif method == "spearman":
                value = pair[left].rank(method="average").corr(pair[right].rank(method="average"))
            else:
                value = pair[left].corr(pair[right])
            output.append({"variable_x": left, "variable_y": right, "method": method,
                           "correlation": value, "n_models": len(pair), "interpretation": "descriptive_not_causal"})
    return pd.DataFrame(output)


def compression_family(method_chain: Any) -> str:
    text = " ".join(json.loads(method_chain) if isinstance(method_chain, str) and method_chain.startswith("[") else [str(method_chain)]).lower()
    matches = [family for family, words in FAMILY_KEYWORDS.items() if any(word in text for word in words)]
    return "combined" if len(matches) > 1 else matches[0] if matches else "baseline"


def family_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy(); data["compression_family"] = data.method_chain.map(compression_family)
    numeric = [column for column in ("f1_score", "parameters", "checkpoint_bytes", "model_time_s", "end_to_end_time_s", "total_energy_kwh") if column in data]
    return data[data.status == "SUCCESS"].groupby("compression_family")[numeric].agg(["count", "mean", "median"]).reset_index()


def paper_tables(frame: pd.DataFrame, robustness: pd.DataFrame | None = None, baseline: str = "A0") -> dict[str, pd.DataFrame]:
    data = model_level(frame)
    metadata = frame.drop_duplicates("model_id").set_index("model_id")
    data["method"] = data.model_id.map(lambda model: metadata.loc[model].get("method_chain"))
    data["precision"] = data.model_id.map(lambda model: metadata.loc[model].get("precision"))
    characteristics = data.rename(columns={"model_id":"Model ID", "method":"Method", "precision":"Precision", "layers":"Layers", "hidden_size":"Hidden size", "intermediate_size":"FFN size", "vocab_size":"Vocabulary", "parameters":"Parameters"})
    characteristics["Checkpoint MB"] = characteristics.get("checkpoint_bytes") / 1024**2
    characteristics = characteristics[[column for column in ("Model ID","Method","Precision","Layers","Hidden size","FFN size","Vocabulary","Parameters","Checkpoint MB") if column in characteristics]]
    baseline_f1 = float(data.loc[data.model_id == baseline, "f1_score"].iloc[0]) if baseline in set(data.model_id) else None
    main = data.copy(); main["delta_f1"] = main.f1_score - baseline_f1 if baseline_f1 is not None else None
    main = main.rename(columns={"model_id":"Model ID", "precision_score":"Precision", "recall_score":"Recall", "f1_score":"F1", "delta_f1":"Delta F1", "gpu_peak_allocated_mb":"Peak VRAM", "model_time_s":"Forward latency", "end_to_end_time_s":"End-to-end latency", "total_energy_kwh":"Energy"})
    main = main[[column for column in ("Model ID","Precision","Recall","F1","Delta F1","Peak VRAM","Forward latency","End-to-end latency","Energy") if column in main]]
    return {"model_characteristics": characteristics, "main_results": main,
            "robustness": robustness_paper_table(robustness) if robustness is not None else pd.DataFrame()}


def robustness_paper_table(table: pd.DataFrame) -> pd.DataFrame:
    selections = {
        "Frequent F1": ("frequency", {">20"}), "Rare F1": ("frequency", {"unseen","singleton","2-5"}),
        "Seen F1": ("seen_unseen", {"seen"}), "Unseen F1": ("seen_unseen", {"unseen"}),
        "Short-mention F1": ("character_length", {"short"}), "Long-mention F1": ("character_length", {"long"}),
        "Low-fragmentation F1": ("subwords", {"1","2"}), "High-fragmentation F1": ("subwords", {"5+"}),
    }
    rows=[]
    for model_id, model in table.groupby("model_id"):
        row={"Model":model_id}
        for title,(dimension,buckets) in selections.items():
            selected=model[(model.dimension==dimension)&(model.bucket.isin(buckets))]
            tp,fp,fn=(selected[column].sum() for column in ("tp","fp","fn"))
            precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
            row[title]=2*precision*recall/(precision+recall) if precision+recall else 0
        rows.append(row)
    return pd.DataFrame(rows)


def joint_task_summary(ner: pd.DataFrame, el: pd.DataFrame, baseline: str = "A0") -> pd.DataFrame:
    """Join task metrics by model without constructing a cross-task score."""
    ner_models=model_level(ner); baseline_rows=ner_models[ner_models.model_id==baseline]
    baseline_f1=None if baseline_rows.empty else float(baseline_rows.f1_score.iloc[0])
    if baseline_f1 is not None: ner_models["ner_delta_f1"]=ner_models.f1_score-baseline_f1
    el_numeric=el.select_dtypes(include="number").columns
    el_models=el.groupby("model_id",as_index=False)[list(el_numeric)].mean()
    selected_ner=[column for column in ("model_id","parameters","checkpoint_bytes","gpu_peak_allocated_mb","total_energy_kwh","f1_score","ner_delta_f1") if column in ner_models]
    selected_el=[column for column in ("model_id","accuracy_at_1","recall_at_25","unseen_accuracy") if column in el_models]
    return ner_models[selected_ner].merge(el_models[selected_el],on="model_id",how="outer")


def export_paper_tables(tables: dict[str, pd.DataFrame], root: str | Path) -> dict[str, Path]:
    root = Path(root); output = {}
    for name, table in tables.items():
        tsv, csv = root / f"paper_{name}.tsv", root / f"paper_{name}.csv"
        table.to_csv(tsv, sep="\t", index=False); table.to_csv(csv, index=False)
        output[f"{name}_tsv"] = tsv; output[f"{name}_csv"] = csv
    return output


def generate_plots(frame: pd.DataFrame, root: str | Path) -> list[Path]:
    """Export ZSTD Parquet plot data and colour-blind-safe vector SVG figures."""
    root = Path(root) / "plots"; root.mkdir(parents=True, exist_ok=True)
    data = model_level(frame); plots = []
    baseline = data[data.model_id == "A0"]
    if not baseline.empty and "checkpoint_bytes" in data and "f1_score" in data:
        data["compression_ratio"] = float(baseline.checkpoint_bytes.iloc[0]) / data["checkpoint_bytes"]
        data["delta_f1"] = data["f1_score"] - float(baseline.f1_score.iloc[0])
    pairs = [("parameters","f1_score"),("checkpoint_bytes","f1_score"),("gpu_peak_allocated_mb","f1_score"),("model_time_s","f1_score"),("end_to_end_time_s","f1_score"),("documents_per_s","f1_score"),("total_energy_kwh","f1_score"),("model_time_s","total_energy_kwh"),("compression_ratio","delta_f1")]
    try: import matplotlib.pyplot as plt
    except ImportError: plt = None
    for x, y in pairs:
        if x not in data or y not in data: continue
        subset = data[["model_id",x,y]].dropna(); parquet = root / f"{y}_vs_{x}.parquet"; subset.to_parquet(parquet,compression="zstd",index=False); plots.append(parquet)
        if plt is not None and not subset.empty:
            figure, axis = plt.subplots(facecolor="white"); axis.scatter(subset[x],subset[y],color="#0072B2")
            for row in subset.itertuples(): axis.annotate(row.model_id,(getattr(row,x),getattr(row,y)))
            axis.set_xlabel(x); axis.set_ylabel(y); axis.grid(False); axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
            figure.tight_layout(); svg=root/f"{y}_vs_{x}.svg"; figure.savefig(svg,format="svg",bbox_inches="tight"); plt.close(figure); plots.append(svg)
    return plots
