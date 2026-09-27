# Scientific analyses and metric interpretation

All core analyses operate on stable IDs from complete snapshots and persist their
results as ZSTD Parquet. Unless `sample_size` is set, Part 2 uses every observation
that is common to every snapshot and has a gold label. Explicit sampling is seeded
and label-stratified. References below point to the local
[`references/references.bib`](../references/references.bib).

## Shared PCA and temporal UMAP

### Scientific question

How does the same population move in a common two-dimensional visual coordinate
system?

### Calculation, input, output

`shared_pca` intersects stable IDs, fits one covariance/eigenvector basis across
all checkpoint matrices, and transforms every checkpoint with that basis.
`fixed_umap` fits one UMAP on concatenated checkpoint matrices and transforms each
matrix with the same fitted object. `aligned_umap` supplies identity relations
between the same observation at consecutive checkpoints. Input is snapshot hidden
states; output is `projection.parquet`.

### Interpretation and caveats

Point movement is comparable because coordinates are shared/aligned. It remains a
low-dimensional visualization: distance and cluster appearance can be distorted.
Never interpret independently fitted UMAPs as physical movement; the arbitrary
rotation/reflection/nonlinear layout differs between fits. UMAP methods require
optional `umap-learn`; PCA uses NumPy. Cost is dominated by covariance/eigendecomposition
or UMAP fitting. Reference: McInnes et al. (2018).

## Representation trajectories

### Scientific question and calculation

How far and how directly does each observation move? Consecutive vectors yield
Euclidean step distance, cosine similarity/distance, and angular change. Each row
also measures distance from pretrained/final, cumulative path length, current/net
displacement, maximum step, and its checkpoint. Output: `trajectories.parquet`.

### Important metrics

* **Euclidean drift** (`distance_previous`) is the L2 norm of one checkpoint step.
* **Cosine drift** (`cosine_distance_previous`) is `1 - cosine_similarity`; it
  emphasizes direction rather than magnitude.
* **Cumulative path length** sums stepwise Euclidean movement.
* **Net displacement** is straight-line distance from pretrained to the current
  point.
* **Path-length/displacement ratio** is cumulative path divided by displacement.
  Values near 1 describe a direct path; larger values indicate detours/instability.
  Zero-displacement cases are reported as 1 by implementation convention.

Group summaries aggregate by checkpoint, layer, gold BIO label, correctness, and
error type. Cost is linear in observations × checkpoints × dimensions once aligned.

## Entity centroid dynamics and representation shifts

### Scientific question

Do label groups become more compact, separated, or stable, and do class means shift
in similar directions?

### Calculation and outputs

Class centroids produce displacement, velocity, acceleration, within-class radius/
variance, nearest competitor, centroid distances, and an inter-class margin in
`centroid_dynamics.parquet`. Groups below `minimum_group_size` carry
`warning: insufficient_group`. `representation_shifts.parquet` stores each class
mean at a checkpoint minus its pretrained mean, the derived vector, and magnitude.

### Caveats and cost

Centroid summaries can hide multimodality; shift vectors are diagnostics, not causal
concept directions. Results depend on label/population consistency. Cost is linear
for centroids plus class-pair comparisons.

## Linear CKA

### Scientific question

How similar are two representation spaces despite isotropic scaling and orthogonal
transformations?

### Calculation and output

Feature-space centered linear CKA compares pretrained-to-each checkpoint, every
checkpoint pair, consecutive states, same layers over time, and cross-layer pairs
within checkpoints. `cka.parquet` stores checkpoint/layer pairs, score, population,
and sample count. A sufficient-statistic accumulator is implemented for streaming
use, although the current main pipeline loads one selected layer across checkpoints
before its comparisons.

CKA is normally in `[0,1]`; larger is more similar. High CKA does not mean individual
observations did not move, and low CKA does not establish worse performance. Cost
uses feature cross-products and can be substantial at large hidden sizes. Reference:
Kornblith et al. (2019).

## SVCCA and PWCCA (optional)

SVCCA performs SVD truncation to configured-in-code 99% retained variance, then
reports mean canonical correlation. PWCCA weights canonical correlations by
activation projection magnitudes. Outputs are `svcca.parquet` and `pwcca.parquet`.
They are disabled by default because repeated decompositions are expensive and can
be unstable with small/rank-deficient samples. References: Raghu et al. (2017),
Morcos et al. (2018).

## Representation geometry

### Scientific question and output

How compact, separated, isotropic, and locally label-consistent is a checkpoint ×
layer population? `geometry.parquet` reports sample counts/warnings and:

* mean intra/inter-class Euclidean and cosine distance;
* silhouette (−1 to 1; higher generally means better class separation);
* Davies–Bouldin (lower is better under its centroid/scatter assumptions);
* Calinski–Harabasz and Fisher discriminant ratio (larger reflects more between-
  class relative to within-class variation);
* exact blockwise kNN label purity (fraction of neighbors sharing the label);
* neighborhood entropy and hubness skewness;
* covariance spectrum, effective rank, participation ratio, dominant-component
  fraction, mean pairwise cosine, and isotropy score.

### Anisotropy metrics

Mean pairwise cosine measures directional alignment. Dominant-component fraction
measures covariance concentration. Effective rank is entropy-based spectral rank;
participation ratio is `(sum eigenvalues)^2 / sum(eigenvalues^2)`. Isotropy score
normalizes effective rank by attainable ambient/sample rank. No single one is
treated as definitive. Exact pairwise class summaries can be expensive; kNN queries
are blocked but exact.

Reference for contextual anisotropy: Ethayarajh (2019).

## Intrinsic dimensionality

`intrinsic_dimension.parquet` contains two estimators. Covariance participation
ratio is a global effective-dimension measure. TwoNN is a local estimator based on
the ratio of second- to first-neighbor distance; the implementation reports the
inverse mean log-ratio. Estimates can be unstable with duplicates, small samples,
or distance concentration. Reference: Facco et al. (2017).

## Nearest-neighbor evolution

### Scientific question

Does an observation's local semantic/label neighborhood reorganize?

### Calculation and output

Exact cosine top-k neighbors are computed in query blocks; the all-pairs matrix is
never persisted. `neighborhoods.parquet` stores neighbor IDs/ranks/distances/
similarities and summary purity, entropy, Jaccard overlap with the previous and
pretrained neighborhoods, and turnover (`1 - previous Jaccard`). Cost is still
quadratic in distance calculations, with memory bounded by `block_size × N`.

## Dataset Cartography

### Scientific question

Which observations are consistently easy, uncertain/variable, or hard across
evaluation snapshots?

### Calculation and output

`training_dynamics.parquet` preserves raw checkpoint histories and summaries of
mean gold-label confidence, population-standard-deviation confidence variability,
correctness frequency, mean entropy/margin, and prediction stability. Regions are
data-derived: high-confidence/high-correctness is `easy_to_learn`, low/low is
`hard_to_learn`, high variability is `ambiguous`, otherwise `middle`; thresholds
are stored. Word/document aggregates are included where metadata permits.

Raw continuous measures should be analyzed rather than treating region names as
ground truth. Reference: Swayamdipta et al. (2020).

## Forgetting events

A forgetting event is `correct → incorrect` after a correct state.
`forgetting.parquet` stores first-correct checkpoint, first checkpoint after which
all remaining states are correct, count/last forgetting event, final correctness,
label changes, and stability after first learning. An observation never correct has
null learning checkpoints. Evaluation snapshots are observations of training
dynamics, not every optimizer step. Reference: Toneva et al. (2019).

## Prediction transitions

`transitions.parquet` aggregates consecutive predicted-label pairs by checkpoint
pair. It reports raw count, probability normalized within source label/checkpoint
pair, correction/degradation counts, and persistence rate. Token transition labels
are BIO predictions; they do not replace the repository's span-level evaluation
logic. Cost is linear in histories.

## Frozen linear probing

### Scientific question

How linearly decodable are entity presence, BIO boundary, and entity type at each
layer/checkpoint?

The NumPy multinomial logistic probe uses stable-ID hashed train/validation/test
assignments shared everywhere. It reports macro precision/recall/F1, accuracy,
test sample count, classes, and warnings in `probing.parquet`. The underlying
encoder is never trained by the probe. High F1 means linear decodability, not that
the model causally uses the information. Reference: Alain and Bengio (2017).

Optional `mdl_probing` uses a deterministic online code with ridge-linear
probabilistic probes and reports code lengths/compression. Reference: Voita and
Titov (2020).

## Parameter drift

### Scientific question

Where did fine-tuning physically alter weights?

Float32 parameter states are compared one tensor at a time against pretrained.
Names are classified into embeddings, Transformer layer attention/FFN/
normalization, classification head, or `other`. `parameter_drift.parquet` reports
absolute L2 delta, baseline-relative delta, identical `update_weight_ratio`, and
parameter count. Relative values are undefined (`NaN`) for a zero baseline norm.
Parameter magnitude does not by itself establish functional importance.

## Performance relationships and layer specialization

Ordinary epoch metrics are joined to internal summaries in
`performance_relationships.parquet`. Spearman rank correlations are reported only
with at least three varying pairs; `causal_interpretation` is explicitly false.
Checkpoint matching maps intermediate snapshot order to available evaluation rows,
so verify alignment for nonstandard evaluation schedules.

`layer_specialization.parquet` outer-joins CKA-to-pretrained, mean probe F1,
participation-ratio intrinsic dimension, anisotropy, and kNN purity by checkpoint/
layer. Correlation and specialization are summaries, not mechanisms.

## Hard-example mining

`hard_examples.parquet` ranks up to 100 observations using available percentile
ranks for cumulative movement, maximum step, entropy, forgetting, variability, and
neighborhood turnover, then joins compact document/token/label metadata. Missing
diagnostics contribute zero to the composite. This is a case-selection aid, not a
calibrated probability of difficulty.

## Attribution helpers (targeted, not pipeline output)

`analyses/optional.py` implements Gradient × Input, Integrated Gradients for a
caller-supplied target score, token occlusion, and basic comprehensiveness/
sufficiency differences. These require a live model and selected cases; enabling
`analyses.attribution` in the corpus-wide CLI is rejected. A manually created
`tables/attribution.parquet` with `token` and `attribution` can be plotted. Saliency
is not automatically faithful. References: Sundararajan et al. (2017), Adebayo et
al. (2018).

## Influence helper (targeted, not pipeline output)

The optional module implements a targeted TracIn-style sum of checkpoint gradient
dot products for caller-provided gradients and returns training-example IDs/scores.
It is not wired to `analyze` and does not automatically create
`influence.parquet`. Reference: Pruthi et al. (2020).

## Topology and adaptation phases

These names exist in configuration/registry but are deliberately rejected by the
pipeline. No persistent-homology or change-point algorithm is implemented in the
current local code. Do not describe their tables as generated outputs. The local
catalogue retains Edelsbrunner and Harer (2010) for a future optional topology
implementation.
