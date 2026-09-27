# Compression artifacts and provenance

[Home](README.md) · [Validation](validation.md) · [Methods](methods/README.md)

An artifact is a new directory containing the transformed representation plus provenance. Output
preparation resolves local source/output paths, rejects source equality, rejects a non-empty output,
creates the directory, and copies `encoding.json` when the source has one. This guarantees
non-destructive transformation, not atomic rollback: a method failure can leave a partial output
that must be removed before retrying.

## File layouts

Normal Hugging Face artifacts (FP16, BF16, magnitude pruning, vocabulary pruning, depth reduction):

```text
OUTPUT/
├── config.json
├── model.safetensors
├── tokenizer_config.json and tokenizer-specific files
├── encoding.json                         # only when source supplied it
├── compression_manifest.json
└── compression_recipe.yaml
```

Dynamic INT8 replaces normal weights with `quantized_model.pt`. SVD writes
`low_rank_state.pt` and `low_rank.json`. ONNX writes `model.onnx`. Tokenizer filenames depend on the
source; vocabulary pruning deliberately removes stale `tokenizer.json` and writes truncated
`vocab.txt`.

## `compression_manifest.json`

| Field | Exact meaning |
|---|---|
| `schema_version` | Currently integer `1`. |
| `source_model` | Source argument converted to string; not a generated artifact ID. |
| `source_hash` | SHA-256 over sorted relative names and bytes for a local directory; hash of the identifier string for a non-local source. |
| `compression_method` | Registry method ID applied in this invocation. |
| `method_chain` | Currently a one-element list. Chaining does not automatically merge an earlier chain. |
| `dtype` | Dtype observed by checkpoint inspection; specialized backends inspect the source. |
| `parameter_count` | Inspected logical parameters or an explicit method override (SVD). |
| `checkpoint_bytes` | Recursive bytes observed at metadata-writing time. For specialized backends this currently comes from source inspection, a known comparability limitation; Phase 2 remeasures artifact-directory bytes. |
| `vocabulary_size` | Config vocabulary size. |
| `hidden_size` | `hidden_size` or architecture alias `dim`. |
| `ffn_size` | `intermediate_size` or `hidden_dim`. |
| `layers` | `num_hidden_layers` or `n_layers`. |
| `heads` | `num_attention_heads` or `n_heads`. |
| `backend` | `transformers`, `pytorch_dynamic_int8`, `pytorch_low_rank`, or `onnx`. |
| `software_versions` | Installed `transformers` and `torch` distribution versions. |
| `scientific_references` | Publication URLs from method metadata. |
| `details` | Method-specific values, e.g. sparsity counts, vocabulary delta, kept layers, rank ratio, task/opset. |

There is no separate `artifact_id` field. Experiment rows use their matrix/model ID as
`artifact_id`. Do not claim otherwise in downstream reports.

## `compression_recipe.yaml`

Generated recipes contain only source `model`, `method.name`, and resolved `output.path`. They do
not necessarily reproduce method-specific CLI options such as magnitude amount, kept tokens,
layers, or rank ratio; those values live in manifest `details`. For strict reproduction, archive
both files and the original explicit command/configuration.

## Source hashes

A local directory hash covers all files, so adding logs changes the hash. A remote identifier hash
does not prove which Hub revision was downloaded. For publication-grade remote provenance, record
a revision/commit outside the current schema.

## Hugging Face compatibility

Normal artifacts are intended for AutoConfig/AutoTokenizer/the corresponding AutoModel class.
Dynamic INT8 and SVD require specialized loaders; canonical `lab.ner.inference.load_model` selects
them from manifest `backend`. ONNX requires an ONNX runtime and is not loadable as a PyTorch
AutoModel. Read [validation](validation.md) before calling an artifact ready.
