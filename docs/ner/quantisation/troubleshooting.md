# Troubleshooting

[Home](README.md) · [CLI](cli.md) · [Validation](validation.md) · [Artifacts](artifacts.md)

| Symptom | Likely cause | Action |
|---|---|---|
| `MODEL` not found / Hub load error | Wrong path, offline uncached identifier, authentication | Use an existing local path; pre-download/authenticate explicitly; run `inspect`. |
| Missing PyTorch/datasets | Installed base package, not NER extra | `uv pip install -e ".[ner]"`. |
| Missing sklearn/FAISS/SentenceTransformers/networkx | NEL extra absent | `uv pip install -e ".[all]"` or `.[nel]`. |
| Output directory is not empty | Non-destructive output guard, possibly partial failed artifact | Inspect/archive it, then remove it or choose a new path. Never point output at source. |
| Unknown method | Used CLI name as registry ID or scaffold ID | Use `methods`; recipe uses `int8-dynamic`, while direct command is `int8`. INT4/static/QAT are unavailable. |
| Unsupported encoder stack | Depth reducer cannot find recognized layer list | Do not force slicing; add/test an architecture adapter or use a trained student. |
| FP16/BF16 CPU forward failure | CPU kernel/hardware lacks dtype support | Store/benchmark on suitable hardware, or record failure; do not silently change conditions. |
| Dynamic INT8 fails on CUDA | Implemented backend is CPU-only | Set controlled experiment `device: cpu`; compare A0 on the same CPU. |
| `onnx` missing | Export dependency absent | Install ONNX in the environment. |
| ONNX validation says specialized runtime | Expected: no ONNX Runtime validator/experiment adapter | Validate with an explicit external provider or implement a documented backend adapter. |
| INT4/static PTQ/QAT/calibration requested | Not implemented | Do not use A4/A5 as executable recipes; implement a scientifically valid method first. |
| Vocabulary command rejects XLM-R | SentencePiece mutation is intentionally analysis-only | Do not renumber IDs; use a separate tokenizer-training/reconciliation experiment. |
| Requested token absent | `--keep-token` not in source vocab | Correct token spelling/tokenizer, then retry into a clean directory. |
| “Selected tokens do not permit tail pruning” | Highest retained ID is already near/end of vocab | Tail-only safe pruning cannot help; do not attempt arbitrary ID removal. |
| Vocabulary validation false | tokenizer/config/embedding rows disagree | Regenerate from source; do not patch only config or vocab. |
| Invalid experiment YAML | Pydantic constraint/unknown field/type error | Compare with [configuration](configuration.md); remember paths resolve from current directory. |
| Experiment model `INVALID` | Missing `encoding.json`, config/tokenizer reload error, unknown A0–C6 ID | Use a saved NER artifact and inspect readiness JSON. |
| Experiment model `UNSUPPORTED` | Backend is not supported by canonical PyTorch worker | Use a matched supported runtime or implement a real adapter. |
| `max_length` failure | YAML differs from saved encoding | Set YAML to artifact `encoding.json`; do not silently change one model. |
| CUDA not detected / requested CUDA error | Driver/runtime/device unavailable | Check PyTorch CUDA, `CUDA_VISIBLE_DEVICES`, and use `cpu` only if it remains the intended protocol. |
| `OOM` | Fixed controlled batch does not fit | Keep OOM as result; use a separate maximum-throughput protocol for model-specific batches. |
| CodeCarbon import/API failure | NER extra absent or incompatible install | Install `.[ner]`; disable energy only if study design permits and record missing values. |
| Energy values null | Hardware/API did not expose component | Keep NA; never fabricate or infer from parameters. |
| Resume reruns a failure | Only exact `SUCCESS` is complete | Fix cause and use `--resume`; expected behavior. |
| Resume uses stale configuration | Resume does not compare payload hashes | Use a new experiment ID or archive and `--force`. |
| `--force` lost logs | It deletes the whole result root | Archive failures before force. |
| `--plots` appears ineffective | Resolved YAML `analysis.plots` currently overrides CLI value | Set `analysis.plots: true` in experiment YAML; recorded inconsistency. |
| Missing compression manifest | A0/original model or incomplete artifact | A0 may legitimately lack it; compressed paths should be regenerated for provenance. |
| Cross-encoder enabled without model | EL Pydantic validation | Set `crossencoder.model` or `enabled: false`. |
| SQ/PQ/IVF/HNSW EL index unsupported | Existing adapter exposes flat indexes only | Use flat index or implement/test a real FAISS adapter; do not relabel flat results. |
