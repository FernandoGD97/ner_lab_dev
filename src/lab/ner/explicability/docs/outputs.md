# Output directory and Parquet schema reference

## Annotated run tree

```text
RUN_DIRECTORY/
├── run_manifest.json                  training/assessment manifest
├── epoch_metrics.parquet              ordinary Trainer metrics, when produced
└── explicability/
    ├── config.yaml                    configuration copied at initialization
    ├── manifest.json                  lifecycle and artifact registry
    ├── SUCCESS                        only after successful `finalize`
    ├── _snapshots/                    temporary; may be deleted after success
    ├── tables/                        permanent analytical/replotting inputs
    ├── figures/
    │   ├── representations/           permanent SVGs
    │   ├── trajectories/
    │   ├── geometry/
    │   ├── dynamics/
    │   ├── neighborhoods/
    │   ├── probing/
    │   ├── attribution/
    │   ├── parameters/
    │   └── composites/
    ├── animations/                    animated SVG; optional MP4
    ├── examples/                      currently reserved
    └── report/index.html              permanent report
```

`tables/`, `figures/`, `animations/`, and `report/` are needed for durable results.
`tables/` plus configuration/manifest are sufficient for ordinary replotting.
`figures/` can be replaced deterministically by `plot`. `_snapshots/` are required
for `analyze`, but not for `plot`, `animate`, or reading the report.

## Manifest

`manifest.json` is extensible JSON, not an array store. It records schema version,
run/model/dataset/split, trigger, snapshots, completed/mandatory analyses, required
outputs, cleanup policy/state, status, storage sizes, analysis parameters/runtime,
visualization status/profile/runtime, figure manifest/report paths, and completed
figure IDs. Snapshot entries include layer/dimension/count/performance metadata but
not embeddings.

## Part 2 tables

Columns below are the columns constructed by the current pipeline. Some mixed
tables use `record_type`, so columns that apply only to another record type are
nullable.

### `projection.parquet`

Major columns: `observation_id`, `checkpoint`, `projection_method`, `x`, `y`
(or additional `component_N` when configured above two dimensions), `layer`,
`representation_level`, `gold_label`, `predicted_label`, `correct`, `confidence`,
`gold_confidence`, `entropy`, `gold_vs_second_best_margin`, `error_type`, sampling
metadata. Coordinates share one PCA basis or an aligned/fixed UMAP system.

### `trajectories.parquet`

Observation records contain `observation_id`, `checkpoint`, `layer`, labels,
correctness/error type, `distance_previous`, `cosine_similarity_previous`,
`cosine_distance_previous`, `angular_change_radians`, `distance_pretrained`,
`distance_final`, `cumulative_path_length`, `net_displacement`,
`path_displacement_ratio`, `maximum_single_step`,
`maximum_movement_checkpoint`, and `final_net_displacement`. `record_type` is
`observation` or `group_summary`; group summaries add mean/median metric columns and
`sample_count`.

There is no current file named `representation_drift.parquet`; drift values live in
`trajectories.parquet`.

### `centroid_dynamics.parquet`

`class_summary` rows include `checkpoint`, `class_label`, `sample_count`, `centroid`,
`displacement_pretrained`, `displacement_previous`, `centroid_velocity`,
`centroid_acceleration`, `within_class_radius`, `within_class_variance`,
`nearest_competing_class`, `nearest_centroid_distance`, `inter_class_margin`,
`warning`, and `layer`. `centroid_pair` rows use `class_a`, `class_b`, and
`centroid_distance`.

### `cka.parquet`, `svcca.parquet`, `pwcca.parquet`

Columns: `checkpoint_a`, `checkpoint_b`, `layer_a`, `layer_b`,
`sample_population`, `sample_count`, and respectively `cka_score`, `svcca_score`,
or `pwcca_score`. CKA includes cross-layer, same-checkpoint records when multiple
layers exist.

### `geometry.parquet`

One checkpoint/layer/representation-level row with `sample_count`, `class_count`,
`warning`, `effective_rank`, `participation_ratio`, `mean_pairwise_cosine`,
`dominant_component_fraction`, `isotropy_score`, `covariance_spectrum`, and—when
class/sample conditions allow—`fisher_discriminant_ratio`, `calinski_harabasz`,
`davies_bouldin`, `knn_label_purity`, `neighborhood_entropy`, `hubness_skewness`,
`silhouette`, and intra/inter-class Euclidean/cosine distances.

### `intrinsic_dimension.parquet`

Columns: `estimator` (`participation_ratio` or `twonn`), `estimate`, `sample_count`,
`parameter`, `checkpoint`, `layer`, and `representation_level`.

### `neighborhoods.parquet`

`neighbor` records contain `checkpoint`, `observation_id`, `neighbor_id`, `rank`,
`distance`, `similarity`, `same_label`, and `layer`. `summary` records contain `k`,
`label_purity`, `neighborhood_entropy`, `jaccard_previous`, `jaccard_pretrained`,
and `neighborhood_turnover`.

### `training_dynamics.parquet`

`summary` rows contain `observation_id`, `mean_confidence`,
`confidence_variability`, `correctness_frequency`, `mean_entropy`, `mean_margin`,
`prediction_stability`, `sample_count`, data-derived `cartography_region`, and easy/
hard thresholds. `checkpoint_history` rows preserve `checkpoint`, order, labels,
correctness, gold/predicted confidence, entropy, and margin. Both carry
`representation_level` (`token`, and derived `word`/`document` where available).

### `forgetting.parquet`

Columns: `observation_id`, `first_correct_checkpoint`,
`first_stable_learning_checkpoint`, `forgetting_events`,
`last_forgetting_checkpoint`, `final_correct`, `label_changes`,
`stability_after_first_learning`, and `representation_level`.

### `transitions.parquet`

Columns: `checkpoint_from`, `checkpoint_to`, `label_from`, `label_to`,
`transition_count`, `corrections`, `degradations`, `persistence_count`,
`transition_probability`, `persistence_rate`, and `representation_level`.

### `probing.parquet` and `mdl_probing.parquet`

Linear probe columns: `checkpoint`, `layer`, `probe_task`, `seed`, `precision`,
`recall`, `f1`, `accuracy`, `sample_count`, `classes`, and `warning`.
Probe tasks are `entity_vs_non_entity`, `bio_boundary`, and `entity_type`.

MDL rows use `online_code_bits`, `uniform_code_bits`, `compression`, `sample_count`,
`fractions`, and `interpretation: decodability_not_causality`.

### `parameter_drift.parquet`

Columns: `checkpoint`, `layer`, `component`, `delta_norm`, `relative_delta`,
`update_weight_ratio`, and `parameter_count`. Components include embeddings,
attention, FFN, normalization, classification head, and fallback `other`.

### Other analysis tables

* `representation_shifts.parquet`: checkpoint, class label, vector magnitude,
  derived shift vector, sample count, layer.
* `performance_relationships.parquet`: mixed `joined_checkpoint` rows and
  `correlation` rows; correlations contain performance/internal metric names,
  Spearman coefficient, sample count, warning, and `causal_interpretation: false`.
* `hard_examples.parquet`: joined continuous difficulty/movement/forgetting/
  neighborhood diagnostics plus document/token/gold/predicted metadata and
  `hardness_rank_score`.
* `layer_specialization.parquet`: checkpoint/layer outer join of available
  `cka_to_pretrained`, `probe_f1`, `intrinsic_dimension`, `anisotropy`, and
  `knn_purity`.

All Part 2 tables also receive `analysis_seed`, `sample_population`, and
`configured_sample_size`.

## Part 3 tables

### `corpus_adaptation_fingerprint.parquet`

Columns: `metric`, `value`, `median`, `sample_count`, `ci_method`, `ci_level`,
`ci_lower`, `ci_upper`, and `source_table`. Confidence intervals are deterministic
percentile bootstrap summaries.

### `figure_manifest.parquet`

Columns: `figure_id`, deterministic `filename`, `category`, `analysis`,
`source_table`, `profile`, representation level, layer, checkpoints, projection
method, JSON `entity_colors`, JSON `rasterized_artists`, JSON `axis_labels`, run ID,
model, split, palette, seed, and JSON `creation_configuration`.

### Cross-run comparison

`lab explicability compare` writes `cross_run_fingerprints.parquet` outside a run.
It contains fingerprint columns plus `run_id`, `model`, and `seed`.
