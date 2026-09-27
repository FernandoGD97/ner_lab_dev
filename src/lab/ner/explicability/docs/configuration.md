# Configuration reference

Explicability is a top-level argument of the existing `ner.train_model` task.
`lab run` passes the YAML mapping to `train_model()`, which resolves it through
`resolve_config()`. Unknown keys are rejected. No separate configuration file is
accepted by `analyze` or `plot`; they read the copy saved as
`<run>/explicability/config.yaml` during training.

## Top level

| Option | Type | Default | Description |
|---|---|---:|---|
| `explicability.enabled` | boolean | `false` | Add snapshot capture to training. When false, the training path creates no explicability callback or directory. |
| `explicability.snapshots` | mapping | defaults below | Part 1 capture/storage behavior. |
| `explicability.analyses` | mapping | defaults below | Part 2 method selection and numerical settings. |
| `explicability.visualization` | mapping | defaults below | Part 3 export, style, sampling, and animation settings. |

## `snapshots`

| Option | Type | Default | Allowed values / behavior |
|---|---|---:|---|
| `trigger` | string | `evaluation` | `evaluation` captures in the Trainer `on_evaluate` callback; `epoch` captures at `on_epoch_end`. Nested train-metric evaluation is excluded. |
| `split` | string | `validation` | Only `validation` is implemented; any other value is rejected. |
| `embedding_scope` | string | `last_hidden_state` | `last_hidden_state`, `selected_layers`, or `all_hidden_states`. Layer 0 is the embedding output; the last index is the final Transformer layer. |
| `selected_layers` | list of integers | `[]` | Required and non-empty only with `selected_layers`. Negative indices resolve from the end; invalid or duplicate resolved layers fail. |
| `dtype` | string | `float16` | `float16` or `float32` temporary embedding storage. Extraction logits/analytical calculations may use higher precision. |
| `include_all_examples` | boolean | `true` | Must currently remain true. `false` is explicitly rejected; scientific sampling is not performed during capture. |
| `include_all_tokens` | boolean | `true` | If true, include every non-padding input position, including model special tokens. If false, include only positions whose gold label is not the ignore index. |
| `cleanup_policy` | string | `on_success` | `on_success` or `never`. See [storage](storage.md). |
| `batch_size` | integer or null | `null` | Snapshot extraction batch size; null resolves to 8. This is independent of Trainer evaluation batch size. |
| `chunk_size` | integer | `16` | Validated as positive, but the current writer emits one compressed chunk per processed encoded row; this value is not currently consumed by extraction. |
| `disk_safety_margin_bytes` | integer | `2147483648` | Free space reserved before each snapshot (2 GiB). |
| `metadata_overhead_fraction` | float | `0.10` | Multiplier added to the raw embedding estimate for the preflight check. Must be non-negative. |

Parameter states are captured with each snapshot when
`analyses.parameter_drift: true` (the default), and their float32 byte estimate is
added to disk preflight requirements.

## `analyses`

Every analysis is independently configurable. The following core booleans default
to `true`:

| Option | Output |
|---|---|
| `temporal_projection` | `projection.parquet` |
| `trajectories` | `trajectories.parquet` |
| `centroid_dynamics` | `centroid_dynamics.parquet` |
| `cka` | `cka.parquet` |
| `geometry` | `geometry.parquet` |
| `intrinsic_dimension` | `intrinsic_dimension.parquet` |
| `neighborhoods` | `neighborhoods.parquet` |
| `training_dynamics` | `training_dynamics.parquet` |
| `forgetting` | `forgetting.parquet` |
| `transitions` | `transitions.parquet` |
| `probing` | `probing.parquet` |
| `parameter_drift` | `parameter_drift.parquet` |
| `representation_shifts` | `representation_shifts.parquet` |
| `performance_relationships` | `performance_relationships.parquet` |
| `hard_examples` | `hard_examples.parquet` |
| `layer_specialization` | `layer_specialization.parquet` |

Optional/expensive booleans default to `false`:

| Option | Current behavior |
|---|---|
| `svcca` | Implemented and written to `svcca.parquet` when enabled. |
| `pwcca` | Implemented and written to `pwcca.parquet` when enabled. |
| `mdl_probing` | Implemented online-code probe, written to `mdl_probing.parquet`. |
| `attribution` | Targeted helper functions exist, but the corpus-wide `analyze` command rejects it because case/model inputs are required. No automatic table is produced. |
| `influence` | Targeted TracIn-style helper exists; corpus-wide `analyze` rejects it. |
| `topology` | Registry/config placeholder only; corpus-wide `analyze` rejects it and no topology implementation is called. |
| `adaptation_phases` | Registry/config placeholder only; corpus-wide `analyze` rejects it. |

Numerical settings:

| Option | Type | Default | Allowed values / meaning |
|---|---|---:|---|
| `projection_method` | string | `shared_pca` | `shared_pca`, `fixed_umap`, `aligned_umap`; UMAP methods require optional `umap-learn`. |
| `projection_dimensions` | integer | `2` | Positive number of coordinates. The Part 3 catalogue expects `x`/`y`, so use at least 2 for standard plots. |
| `neighbors_k` | integer | `10` | Positive top-k for exact blockwise nearest neighbors. |
| `probe_seed` | integer | `42` | Hash seed for fixed probe splits. |
| `probe_train_fraction` | float | `0.7` | Strictly between 0 and 1. |
| `probe_validation_fraction` | float | `0.15` | In `[0,1)`; train + validation must be less than 1, leaving a test set. The current probe trains on train and reports test metrics; validation membership is persisted by the deterministic split logic but not used for tuning. |
| `probe_max_iterations` | integer | `200` | Gradient-descent iterations for the NumPy multinomial linear probe. |
| `probe_learning_rate` | float | `0.1` | Probe optimizer step size. |
| `minimum_group_size` | integer | `3` | Groups below this threshold receive warnings/limited geometry; minimum accepted value is 2. |
| `block_size` | integer | `1024` | Positive query block size for nearest-neighbor/distance work. |
| `workers` | integer | `1` | Must be positive. The current orchestrator is sequential and does not consume this setting. |
| `sample_size` | integer or null | `null` | Null uses every stable gold-labelled observation. A number ≥2 activates explicit deterministic stratified analytical sampling. |
| `sample_seed` | integer | `42` | Seed for analytical sampling and stochastic projection configuration; recorded in outputs. |

There is no configurable `representation_levels` key. Core representation matrices
are analyzed at token level; `training_dynamics`, `forgetting`, and `transitions`
also derive word/document aggregations when stable metadata is available.

## `visualization`

| Option | Type | Default | Allowed values / meaning |
|---|---|---:|---|
| `profile` | string | `paper` | `paper` or `presentation`. |
| `static_format` | string | `svg` | Must be `svg`; SVG is canonical. |
| `pdf`, `png` | boolean | `false` | Direct aliases for optional derivatives. |
| `optional_formats.pdf`, `.png` | boolean | `false` | Nested YAML form of the same settings; requires optional `cairosvg`. |
| `palette` | string | `okabe_ito` | `okabe_ito`, `tol_bright`, or `colorbrewer_dark2`. |
| `background` | string | `white` | Must resolve to `white`. |
| `grid` | boolean | `false` | `true` is rejected. |
| `top_spine`, `right_spine` | boolean | `false` | Direct forms; `true` is rejected. |
| `spines.top`, `.right` | boolean | `false` | Nested forms of the same settings. |
| `smart_legend` | boolean | `true` | Stored/configured; figures currently use deterministic external legends directly. |
| `outside_legend_when_dense` | boolean | `true` | Stored/configured; external placement is the current behavior. |
| `legend.smart_position`, `.outside_when_dense` | boolean | `true` | Nested forms. |
| `dense_scatter_rasterized` | boolean | `true` | Stored in configuration. The native SVG renderer currently controls size by deterministic decimation/opacity rather than embedding a raster artist. |
| `rasterization.dense_scatter` | boolean | `true` | Nested form. |
| `raster_dpi` / `rasterization.dpi` | integer | `600` | Minimum accepted value 72; used for PNG derivative width, not SVG. |
| `visual_sample_size` | integer | `20000` | Minimum 100. Visual-only deterministic limit for dense projection/animation rendering. |
| `visual_sample_seed` | integer | `42` | Stable visual-only sample seed. |
| `animation` | boolean | `true` | Full `plot` pass generates animated SVG unless `--no-animation`. |
| `animation_fps` | integer | `2` | Positive MP4 frame rate and inverse SVG frame duration. |
| `animation_gif` | boolean | `false` | Present in configuration but not consumed; GIF output is not implemented. |

## Examples

### Minimal

```yaml
explicability:
  enabled: true
```

This uses evaluation triggers, the final hidden state, float16, all core analyses,
paper SVGs, animations, and `on_success` cleanup.

### Recommended representation study

```yaml
explicability:
  enabled: true
  snapshots:
    trigger: evaluation
    embedding_scope: selected_layers
    selected_layers: [0, 4, 8, -1]
    dtype: float16
    cleanup_policy: on_success
  analyses:
    projection_method: shared_pca
    neighbors_k: 15
    block_size: 1024
    sample_size: null
  visualization:
    profile: paper
    palette: okabe_ito
    visual_sample_size: 20000
```

### Advanced/full

```yaml
explicability:
  enabled: true
  snapshots:
    trigger: evaluation
    split: validation
    embedding_scope: all_hidden_states
    dtype: float32
    include_all_examples: true
    include_all_tokens: true
    cleanup_policy: never
    batch_size: 4
    disk_safety_margin_bytes: 10737418240
    metadata_overhead_fraction: 0.15
  analyses:
    svcca: true
    pwcca: true
    mdl_probing: true
    neighbors_k: 25
    sample_size: null
    sample_seed: 17
  visualization:
    profile: presentation
    static_format: svg
    optional_formats:
      pdf: false
      png: false
    palette: tol_bright
    animation: true
    animation_fps: 3
```

This can be very expensive: all layers × all snapshots × full population, float32,
plus parameter states. Use `cleanup_policy: never` only with deliberate storage
planning.
