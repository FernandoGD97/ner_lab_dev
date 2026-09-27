# Artifact validation

[Home](README.md) · [Artifacts](artifacts.md) · [Troubleshooting](troubleshooting.md)

```bash
lab quantisation validate ARTIFACT [--source SOURCE]
```

The command prints a `ValidationReport` JSON with these booleans:

| Field | Check |
|---|---|
| `config_loads` | `AutoConfig.from_pretrained` succeeded (normal backend). |
| `tokenizer_loads` | `AutoTokenizer.from_pretrained` succeeded. |
| `model_loads` | Appropriate normal/specialized model loader succeeded. |
| `task_head_preserved` | With `--source`, resolved source/artifact task strings match; without it this is true after load. Specialized paths currently report true after load. |
| `label_mappings_preserved` | With `--source`, both `label2id` and `id2label` match; without it true. Specialized paths currently report true after load. |
| `vocabulary_consistent` | `len(tokenizer) == model.get_input_embeddings().num_embeddings`. |
| `forward_succeeds` | Tokenizes literal `"validation input"` and executes one no-grad forward. |
| `outputs_finite` | All logits or last hidden-state values are finite. |
| `shapes_preserved` | Current implementation checks batch dimension equals one; it does not compare all source tensor dimensions. |
| `valid` | Logical `all()` of the fields above. |

There are no PASS/FAIL/WARNING status strings. A fully completed report has `valid: true`; loading or
forward failures generally raise before a report is returned.

## Backend behavior

* `transformers`: reloads config, tokenizer, and inferred AutoModel class.
* `pytorch_dynamic_int8`: loads `quantized_model.pt` on CPU and smoke-runs it.
* `pytorch_low_rank`: rebuilds source architecture from `low_rank.json`, reapplies factorization,
  loads state, and smoke-runs it.
* `onnx`: raises `ValueError` instructing the caller to use an installed ONNX Runtime provider.
  No ONNX validation CLI currently exists.

## Recommended use

```bash
lab quantisation validate artifacts/a1-fp16 --source models/original
```

Always pass `--source` when label/task preservation matters. Validation checks technical
loadability, not predictive equivalence; run the controlled benchmark next.

## Troubleshooting examples

* **Vocabulary mismatch:** inspect `config.vocab_size`, tokenizer length, and embedding rows. Remove
  partial output and regenerate; never hand-edit only one component.
* **HF reload failure:** verify normal artifact has weights/config and specialized artifact has a
  correct manifest backend.
* **ONNX unsupported message:** expected; use an ONNX Runtime-specific external check.
* **Finite/forward failure:** confirm device supports dtype and task selection is correct.
