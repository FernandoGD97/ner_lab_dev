# Visualization, animation, and report

## Design and profiles

The renderer is SVG-native and does not require Matplotlib. Every static figure is
written as SVG; PDF/PNG are optional CairoSVG derivatives. Text remains searchable,
and axes, labels, arrows, points, and legends are vector elements. SVG has no
meaningful DPI; `raster_dpi` applies only to PNG conversion.

The `paper` profile uses 7.2 × 4.3 inches, 9-point labels, restrained lines, and
compact markers. `presentation` uses 12.8 × 7.2 inches (16:9), 18-point labels,
stronger strokes, and larger markers. The plotted values do not change.

All figures have white backgrounds, no grid, hidden top/right spines, subtle left/
bottom axes, and Arial/Helvetica/sans-serif text. Legends are deterministic and
external. Okabe–Ito is default; Paul Tol Bright and ColorBrewer Dark2 are available.
The outside class is gray. Continuous color uses viridis/cividis, never rainbow.

## Temporal alignment: a necessary caveat

An independently fitted UMAP can arbitrarily rotate, reflect, or nonlinearly
rearrange each checkpoint. Comparing such panels as if coordinates represented
physical motion is invalid.

This implementation offers:

* **shared PCA**: one basis fitted across all aligned checkpoint matrices;
* **fixed UMAP**: one UMAP fitted to concatenated states, then one transform used;
* **AlignedUMAP**: explicit identity relations connect the same observations.

It does not implement Procrustes alignment. Part 3 reads the coordinates Part 2
already persisted; it never refits a projection. All temporal panels use global
coordinate limits. Even aligned projections remain visual summaries, not proof of
mechanistic movement.

## Dense points and visual sampling

Projection figures can deterministically limit the displayed outside background
and very large entity populations by stable-ID hash, seed, and label-stratified
quota. This changes only rendering. It does not modify `projection.parquet` or any
analysis. The visual seed/limit and color mapping are recorded in the figure
configuration/manifest. The current native renderer uses decimation, small points,
and opacity; `dense_scatter_rasterized` is stored but does not currently rasterize
an artist into SVG.

## Figure catalogue

Figures are generated conditionally when their source table/columns exist.

| Figure ID/type | Source | Display and interpretation |
|---|---|---|
| `01_performance_training` | `performance_relationships.parquet` | NER metric versus checkpoint; x is checkpoint, y is the preferred available F1-like metric. |
| `02_shared_pca_evolution` | `projection.parquet` | Shared-PCA checkpoint small multiples, entities plus controlled `O` background. Global x/y limits and entity colors. |
| `03_aligned_umap_evolution` | `projection.parquet` | Emitted only for `aligned_umap`; entity-only aligned panels. |
| `04_temporal_embedding_small_multiples` | `projection.parquet` | Entity-only temporal representation panels. |
| `embedding_correctness_evolution` | `projection.parquet` | Correct/incorrect/unlabelled view; incorrect points also receive a dark edge. |
| `embedding_uncertainty_confidence` | `projection.parquet` | Final-checkpoint confidence encoded with cividis. Do not conflate confidence with calibration. |
| `05_entity_centroid_trajectories` | `projection.parquet` | Projected class centroids connected with directional arrows. These are centroids in projected coordinates. |
| `selected_representation_trajectories` | projection + trajectories | Up to 20 highest cumulative-path observations, with start/end markers/arrows. Selection is intentionally not population-wide. |
| `06_representation_drift`, `distance_from_final`, `cumulative_path_length` | `trajectories.parquet` | Layer-grouped temporal lines for complementary movement metrics. |
| `07_cka_to_pretrained` | `cka.parquet` | CKA against pretrained by layer/checkpoint. |
| `08_cka_layer_similarity` | `cka.parquet` | Final available same-checkpoint layer × layer heatmap. |
| `09_cka_checkpoint_similarity` | `cka.parquet` | Highest-layer checkpoint × checkpoint heatmap. |
| `cka_consecutive_change` | `cka.parquet` | CKA between adjacent checkpoint order by layer. |
| `10_layer_adaptation_map` and `layer_adaptation_*` | `layer_specialization.parquet` | Layer × checkpoint heatmaps for available CKA, probe F1, intrinsic dimension, anisotropy, and kNN purity. |
| `11_intrinsic_dimension` | `intrinsic_dimension.parquet` | Participation-ratio estimate versus checkpoint by layer. |
| `12_effective_rank`, `13_anisotropy`, geometry figures | `geometry.parquet` | Layer trajectories for rank, cosine anisotropy, silhouette, Fisher ratio, participation ratio, and intra/inter-class distance. |
| `14_entity_compactness`, `15_entity_separation` | `centroid_dynamics.parquet` | Within-class radius and nearest-centroid distance by class/checkpoint. |
| `16_knn_purity` | `geometry.parquet` | Local label purity by layer/checkpoint. |
| `17_neighborhood_stability` | `neighborhoods.parquet` | Mean overlap with pretrained neighborhood over time. |
| `semantic_neighborhood_change` | `neighborhoods.parquet` | First selected observation's pretrained/middle/final top neighbors and similarity; IDs are shown because surface text is not in the Part 2 table. |
| `18_dataset_cartography` | `training_dynamics.parquet` | x = mean gold confidence; y = confidence variability; color = data-derived region. |
| `selected_learning_timeline` | `training_dynamics.parquet` | One observation's gold confidence with correct/incorrect markers over checkpoints. |
| `19_forgetting_events` | `forgetting.parquet` | ECDF of token forgetting counts; sample size is annotated. |
| `20_first_learning` | `forgetting.parquet` | ECDF of first-correct checkpoint order among learned observations. |
| `21_prediction_transitions` | `transitions.parquet` | Source-label × destination-label heatmap of normalized transition probability. No Sankey renderer is currently implemented. |
| `22_probing_layer_epoch`, `probing_*` | `probing.parquet` | Layer × checkpoint F1 heatmaps for entity type and each probe task. Decodability is not causality. |
| `23_parameter_drift` | `parameter_drift.parquet` | Architecture component × checkpoint heatmap of relative delta. |
| `24_performance_vs_internal_change` | relationships/CKA/trajectory/geometry tables | A/B/C/D-style stacked panels with independent y-scales and shared checkpoint semantics; avoids misleading dual axes. |
| `25_hard_example_panels` | `hard_examples.parquet` | Compact ranked diagnostic table. It cannot display original sentence text unless that text was persisted elsewhere. |
| `26_corpus_adaptation_fingerprint` | generated fingerprint table | Horizontal summary bars; values are not placed on a common scientific unit scale, so compare within metric, not bar lengths across heterogeneous metrics. |
| `attribution_selected_case` | manually supplied `attribution.parquet` | Token contribution boxes, blue positive/orange negative, normalized within the selected case. |

## Composite and cross-run outputs

The fingerprint table contains means, medians, sample counts, and deterministic
95% percentile-bootstrap intervals for available drift/geometry/dimension/
parameter/forgetting metrics. `lab explicability compare` creates a run × metric
heatmap and combined table. The current comparison does not calculate pairwise
hypothesis tests or multiple-testing corrections.

## Animation

The animation uses `projection.parquet`, filters token rows and the highest layer,
checks identical sorted IDs in every frame, fixes coordinate limits/colors, and
applies deterministic visual sampling. `plot` writes animated SVG unless disabled;
`animate` regenerates it independently. Optional MP4 needs CairoSVG, ImageIO, and a
working H.264 codec. GIF is not implemented.

## Figure manifest and overwriting

Every plotting pass writes or merges `tables/figure_manifest.parquet`. Filenames
are deterministic, so regeneration atomically replaces the SVG at the same path.
A full run rebuilds the manifest from applicable figures; a filtered run replaces
selected IDs while preserving other rows and marks visualization partial.

The manifest records source table, run/model/split, profile, representation level,
layers/checkpoints, projection method, palette, seed, JSON entity-color mapping,
axis labels, rasterized-artists list, and full creation configuration.

## HTML report

`report/index.html` is generated on every successful plot call. It embeds canonical
SVGs using relative paths, groups them by scientific topic, displays run metadata,
adds conservative captions/method caveats/references, links animation files, and
provides a client-side text filter for figures. It is self-contained except for the
linked local SVG/animation artifacts and does not use Plotly.

## Scientific cautions

* Projection appearance is not evidence of causal organization.
* Axis truncation is avoided by padded full data ranges, but projected axes have no
  inherent semantic unit.
* Probe heatmaps show decodability only.
* Parameter movement is not parameter importance.
* Performance correlations are descriptive, not causal.
* Attribution colors require separate faithfulness analysis.
