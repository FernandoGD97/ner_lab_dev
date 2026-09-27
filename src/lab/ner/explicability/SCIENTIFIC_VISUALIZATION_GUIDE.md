# Scientific Visualization Guide

## Architecture and regeneration

Part 3 reads only permanent files in `tables/`; ordinary figure regeneration does
not access `_snapshots/` or recompute Part 2 analyses. The visualization package is
split into style, palette, layout, export, legends, annotations, validation,
domain-specific figures, composites, animation, catalogue orchestration, and HTML
reporting modules.

```bash
lab explicability plot RUN --profile paper
lab explicability plot RUN --profile presentation
lab explicability plot RUN --analysis cka
lab explicability plot RUN --figure 18_dataset_cartography
lab explicability plot RUN --no-animation
lab explicability animate RUN
lab explicability compare OUTPUT_DIR RUN_A RUN_B --profile paper
```

Selective regeneration merges records into the existing figure manifest and is
marked partial; it cannot unlock cleanup. A complete unfiltered plotting pass must
validate before finalization.

## SVG, typography, and profiles

SVG is always the canonical static format. The native SVG renderer keeps text,
axes, annotations, centroids, lines, and arrows as vector objects, with searchable
text in the widely available Arial/Helvetica/sans-serif stack. SVG has physical
dimensions but no scientifically meaningful DPI. PDF/PNG are optional derivatives
through `cairosvg`; raster DPI applies only to raster derivatives.

The paper profile uses a compact double-column 7.2 × 4.3 inch canvas and 9-point
typography. The presentation profile uses a 12.8 × 7.2 inch 16:9 canvas and
18-point typography. Values, filtering, and scales are unchanged across profiles.
Reusable single-column, double-column, wide, square, and 16:9 presets are defined
centrally.

Every plot uses a white background, no grid, subtle left/bottom axes, hidden
top/right spines, restrained strokes, and external legends. Multi-panel figures
use stable A/B/C/D labels.

## Accessibility and legends

Okabe–Ito is the default categorical palette; Paul Tol Bright and ColorBrewer
Dark2 are supported. Labels are sorted deterministically and the outside class is
neutral gray. Entity-color mappings are written into `figure_manifest.parquet`.
Continuous values use viridis or cividis, never rainbow/jet. Important distinctions
also use labels, markers, or line forms rather than color alone. Legends are drawn
outside data axes with stable ordering, avoiding point occlusion and clipping.

## Temporal figures and dense scatter

Temporal panels consume the shared/fixed projection from `projection.parquet` and
use one global pair of coordinate limits. They never refit or independently rescale
checkpoints. Entity-only, controlled-background, correctness, and uncertainty views
are rendering choices, not changes to the analytical population.

Outside-background decimation is deterministic and affects visualization only.
The seed and limit are stored in figure metadata. The analytical Parquet files are
never modified. For dense point clouds, small transparent points are used; optional
raster derivatives use the configured 600 DPI while axes and text remain vector in
the canonical SVG.

## Catalogue and interpretation

The core catalogue covers performance, projection evolution, centroid trajectories,
representation drift, four CKA views, layer adaptation, intrinsic dimension,
effective rank, anisotropy, compactness, separation, kNN purity, neighborhood
stability, Dataset Cartography, forgetting, first learning, prediction transitions,
probing, parameter drift, performance/internal-change composites, hard examples,
and the corpus-adaptation fingerprint. Figures are generated only when their
source Parquet schema is available.

Projection shows visual geometry; CKA/drift show representation change; centroid
and geometry metrics show class organization; neighbors show local reorganization;
Cartography and forgetting show example difficulty; probes show decodability;
parameter drift shows weight-space change. None alone establishes causality.

The report gives factual descriptions, caveats, and local literature references.
Draft captions state population, source, checkpoint scope, and non-causal status.

## Animation, report, and finalization

`animations/temporal_embedding_evolution.svg` is a fixed-coordinate, stable-identity
animation regenerated from projection Parquet. MP4 is additionally emitted when
the optional `cairosvg`, `imageio`, and video codec stack is installed. The
responsive `report/index.html` embeds canonical SVGs and links supplementary
animations; no notebook or web server is required.

Before finalization, validation checks SVG presence, size, XML parsing, dimensions,
style metadata, required source Parquet readability/non-emptiness, and figure
manifest membership. Failure occurs before `SUCCESS`, so `_snapshots/` is retained.
