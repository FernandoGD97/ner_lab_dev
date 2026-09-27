# Configuration reference

[Home](README.md) · [CLI](cli.md) · [Experiments](experiments.md)

Three independent YAML schemas exist: a one-step compression recipe, a controlled NER experiment,
and an EL experiment. All use `yaml.safe_load` followed by Pydantic v2 validation. Relative paths
remain relative to the process working directory.

## Compression recipe

```yaml
model: models/original
method:
  name: fp16
output:
  path: artifacts/a1-fp16
status: READY
```

| Field | Required/default | Notes |
|---|---|---|
| `model` | required string | Source passed to the method. |
| `method.name` | required | Registry ID, not necessarily CLI command (`int8-dynamic`, not `int8`). |
| `method.mode` | optional `null` | Parsed but currently excluded before `apply`; it has no effect. |
| Other `method.*` | allowed | Forwarded as keyword arguments, e.g. `amount`, `layers`, `rank_ratio`, `task`. |
| `output.path` | required path | Replaced by CLI `--output` when present. |
| `status` | `READY` | Informational; no enum/runnability enforcement in this schema. |

Precedence is **Pydantic defaults → YAML → CLI `--output` only**. There are no generic CLI method
overrides for `run-recipe`. The A0–C6 files in `src/lab/ner/quantisation/recipes/` are planning
matrix entries; combined names such as `distillation+int8-dynamic` are not registered executable
pipelines.

## Controlled NER experiment

```yaml
experiment:
  id: biomedical_ner_compression_v1
  seed: 42

dataset:
  path: data/test.parquet
  split: test
  reference: null

inference:
  batch_size: 16
  max_length: 256
  device: cpu
  warmup_batches: 10
  repetitions: 5
  pad_to_multiple_of: 8
  number_of_workers: 0

models:
  - id: A0
    path: models/original
    runtime: pytorch
    backend: null
    execution_provider: null
    representation: huggingface
  - id: A1
    path: artifacts/a1-fp16
    runtime: pytorch
    backend: null
    execution_provider: null
    representation: huggingface
  - id: A3
    path: artifacts/a3-int8
    runtime: pytorch
    backend: pytorch_dynamic_int8
    execution_provider: null
    representation: huggingface

energy:
  enabled: false
  modes: [end_to_end]
  min_duration_seconds: 60
  measure_power_secs: 1

statistics:
  bootstrap_iterations: 10000
  confidence_level: 0.95
  seed: 42

analysis:
  training_frequency_source: null
  seen_definition: normalized_mention_string
  plots: false

output:
  directory: results

protocol: controlled
```

### Field dictionary

| Section.field | Default/constraint | Meaning |
|---|---|---|
| `experiment.id` | required; letters/digits/`_.-` | Directory suffix and result identity. |
| `experiment.seed` | `42` | Counterbalanced execution order seed. |
| `dataset.path` | required | Canonical parquet or input accepted by `read_documents`. |
| `dataset.split` | `test` | Recorded label; it does not select a parquet partition. |
| `dataset.reference` | `null` | Optional separate gold TSV/parquet. |
| `inference.batch_size` | `16`, ≥1 | Fixed controlled batch. |
| `inference.max_length` | `256`, ≥2 | Must equal saved `encoding.json` maximum or the run fails. |
| `inference.device` | `auto` | PyTorch device string; dynamic INT8 ultimately requires CPU. |
| `warmup_batches` | `10`, ≥0 | Direct model batches before peak reset/timing. |
| `repetitions` | `5`, ≥1 | Fresh subprocess runs per model. |
| `pad_to_multiple_of` | `8`, ≥1 or null | Data-collator padding and padded-token accounting. |
| `number_of_workers` | exactly `0` | Canonical helper fixes zero; other values are rejected. |
| `models[].id` | required, unique | Must be one of the A0–C6 matrix IDs. |
| `models[].path` | required | Artifact; must include `encoding.json`. |
| `runtime` | `pytorch` | Explicit result factor. Controlled models must match. |
| `backend` | `null` | Override manifest backend; usually leave null. |
| `execution_provider` | `null` | Explicit factor; controlled models must match. |
| `representation` | `huggingface` | Recorded descriptive factor. |
| `models[].batch_size` | null | Only maximum-throughput protocol may set it. |
| `energy.enabled` | false | Imports CodeCarbon only when enabled. |
| `energy.modes` | `[end_to_end]` | Accepted: `end_to_end`, `model_only`. |
| `min_duration_seconds` | `60` | Repeat fixed workload until elapsed. |
| `measure_power_secs` | `1` | CodeCarbon sampling period. |
| `statistics.*` | 10000 / .95 / 42 | Post-hoc paired bootstrap settings. |
| `analysis.training_frequency_source` | null | Training annotation TSV/parquet, never test data. |
| `analysis.seen_definition` | `normalized_mention_string` only | Recorded long-tail definition. |
| `analysis.plots` | false | Post-hoc optional plotting. |
| `output.directory` | required | Parent; actual root adds `experiment.id`. |
| `protocol` | `controlled` | `controlled` or `maximum_throughput`. |

Unknown top-level and model fields are forbidden. Some nested models use Pydantic's default extra
behavior (ignore), so do not rely on typos being rejected everywhere. Controlled mode requires A0,
forbids per-model batches, and requires a single runtime/provider. `maximum_throughput` merely
permits per-model batch sizes; it does not automatically search for the best batch.

CLI run flags do not override YAML measurement fields. `--dry-run`, `--resume`, and `--force`
control orchestration only. Every run writes a fully default-expanded `resolved_experiment.yaml`.

## Entity Linking experiment

```yaml
experiment: {id: el_v1, seed: 42}
task:
  type: entity_linking
  mentions: data/el-test.tsv
  gazetteer: data/terminology.tsv
  training_mentions: data/el-train.tsv
retrieval:
  candidate_k: [200, 100, 50, 25, 10]
  method: matrix
  index_type: flat_ip
  distance_metric: cosine
  method_kwargs: {}
biencoder:
  id: A0
  model: null
  embedding_dimension: null
crossencoder:
  id: reranker
  enabled: false
  model: null
  batch_size: 16
analysis: {seen_unseen: true, frequency_buckets: true}
energy: {enabled: false, modes: [end_to_end], min_duration_seconds: 60, measure_power_secs: 1}
output: {directory: results}
repetitions: 5
```

Positive unique K values are required. `flat_l2` requires `l2`; `flat_ip` accepts `ip`/`cosine`.
Only flat indexes are currently executable. Enabling the cross-encoder requires a model path.
`biencoder.embedding_dimension` records a real model/projection output dimension; it does not create
a projection.
