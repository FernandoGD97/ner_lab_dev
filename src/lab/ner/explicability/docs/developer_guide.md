# Developer guide

## Architecture at a glance

```text
lab console script
  ↓ argparse handlers in lab.cli (lazy imports)
  ├─ lab run YAML
  │    ↓ ner.train_model → training.train
  │    ↓ ExplicabilityCallback
  │    ↓ capture_snapshot → rank shards/chunked NPZ + metadata Parquet
  │
  ├─ explicability analyze
  │    ↓ SnapshotCollection + AnalysisConfig
  │    ↓ explicit pipeline orchestration
  │    ↓ ZSTD Parquet + manifest update
  │
  ├─ explicability plot
  │    ↓ TableCache + explicit _emit_catalog
  │    ↓ native SVG + figure_manifest + HTML report
  │    ↓ figure/source/style validation
  │
  └─ explicability finalize
       ↓ lifecycle/output/figure/manifest validation
       ↓ SUCCESS
       ↓ optional snapshot cleanup
```

## CLI registration

Despite references to “Typer” in some task descriptions, the local implementation
is argparse. `pyproject.toml` installs `lab = lab.cli:main`. `build_parser()` creates
the root parser and nested `explicability` parser; handler functions lazily import
the heavy packages. There are no decorators, Typer callback, or command aliases.

When adding help, preserve lazy imports and the layering check: `lab.cli` must not
statically import `lab.ner` or PyTorch. Expected user exceptions should generally
be `ValueError`, `FileNotFoundError`, or `TypeError` if they should become concise
exit-status-2 messages under the current `main()`.

## Configuration flow

The YAML runner passes the top-level `explicability` mapping to `train_model()`,
then `train()`. `resolve_config()` creates frozen `SnapshotConfig`,
`AnalysisConfig`, and `VisualizationConfig` dataclasses and rejects unknown keys.
`initialize_run()` writes `asdict(config)` to `config.yaml`. Later CLI stages read
that saved file, so changes to the original training YAML have no effect unless the
saved file is deliberately updated.

Some accepted fields are currently reserved/not consumed (`snapshots.chunk_size`,
`analyses.workers`, visualization smart-legend/rasterization switches, and
`animation_gif`). Maintain documentation when these become operational.

## Training integration and hooks

`training.trainer.train()` always resolves configuration. It imports/constructs
explicability objects only when enabled. The callback holds the existing Trainer,
tokenizer, and validation DataFrame; it does not create a second training/evaluation
pipeline.

`capture_snapshot()`:

1. resolves rank/world size;
2. resolves hidden-state tuple indices;
3. estimates disk;
4. takes deterministic rank-strided encoded rows;
5. runs inference batches with hidden states enabled;
6. writes each row incrementally via `ChunkedArrayWriter`;
7. writes observation metadata/shard summaries;
8. optionally writes parameter state on rank zero;
9. barriers, checks every shard, writes snapshot metadata, and atomically publishes.

Any exception returns model training mode in a `finally` block and leaves incomplete
artifacts rather than publishing validity.

## Storage abstractions

`ChunkedArrayWriter` owns temporary directory/marker/index/chunk behavior.
`iter_chunks()` refuses arrays without `COMPLETE`. `write_table()` optionally stable-
sorts and atomically writes PyArrow ZSTD Parquet; `validate_table()` opens Parquet
metadata. `atomic_json()` provides manifest/snapshot atomic replacement.

Stable IDs use SHA-256 over an unambiguous JSON array of kind/components. Do not
replace this with Python's process-randomized `hash()`.

## Analysis architecture

`analyses/registry.py` is a discoverability registry of names, output filenames,
and expensive flags, and exposes a decorator for future functions. The current
`run_analyses()` **does not dynamically dispatch registry functions**; it explicitly
orchestrates each enabled method in `pipeline.py`. Consequently, adding a registry
entry alone does not make an analysis run.

`SnapshotCollection` sorts snapshot chronology, validates completion, iterates rank
chunks, loads a layer in stable-ID order, reads metadata, and locates parameter
states. The pipeline intersects IDs, restricts core analysis to gold-labelled
positions, optionally samples explicitly, processes layers, writes tables, and
updates the manifest. Keep expensive matrices scoped to one layer/method and avoid
permanent all-pairs outputs.

## Visualization architecture

There is no dynamic visualization registry. `visualization/catalog.py` contains an
explicit `_emit_catalog()` catalogue. `TableCache` lazily caches Parquet frames;
builders return `SvgFigure` or `(SvgFigure, entity_color_mapping[, metadata])`.
`emit()` selects deterministic paths, exports SVG, and creates a `FigureArtifact`.

The native SVG primitives live in `layout.py`; all new figures must use centralized
`StyleProfile`, palettes, external legends, annotations, and `export_figure()`.
`figure_manifest.parquet` is the single metadata catalogue—do not add per-figure
JSON sidecars.

## Manifest and finalization

Lifecycle status progresses through `collecting`, `training_failed` or
`awaiting_postprocessing`, `analyses_complete`, `visualizations_complete`, and
`success`. Visualization also has `partial`/`complete` status. The manifest lists
mandatory/completed analyses and required relative output paths.

`finalize_run()` is the deletion boundary. Required outputs must be present;
Parquet is opened; snapshots and manifest are validated; and completed Part 3
figures are revalidated. Only after manifest write and `SUCCESS` does cleanup run.
Preserve this ordering.

## Adding an analysis

1. Put numerical code in a focused module under `analyses/`; accept arrays/frames
   and return tidy DataFrames without writing files.
2. Add a boolean/settings to `AnalysisConfig` with validation and a documented
   default.
3. Add its name/output to `analyses/registry.py` and `pipeline.OUTPUTS`.
4. Integrate it explicitly into `run_analyses()`; registry registration alone is
   insufficient today.
5. Define a stable, long-form schema with sample count, checkpoint/layer/
   representation population, method parameters, warning fields, and seed/sampling
   metadata where relevant.
6. Let the common pipeline call `write_table()` and manifest update. Do not write
   CSV or duplicate raw embeddings.
7. If it needs figures, add a source-aware builder and catalogue entry.
8. Add small deterministic unit tests, a pipeline smoke case, and schema assertions.
9. Document question, calculation, interpretation, caveats, and cost here/in
   `analyses.md`.
10. Add only a supported original/reference entry to the local reference catalogue.

Targeted methods needing a live model/case should not be forced into the corpus-wide
snapshot orchestrator. Provide an explicit library API and durable table contract.

## Adding a figure

1. Identify an existing permanent Parquet source; Part 3 must not recompute hidden-
   state analyses.
2. Add a focused builder under `visualization/` that returns `SvgFigure` (and color
   mapping/metadata if applicable).
3. Use `profile()`, `Axes`, palette helpers, external legends, and shared limits.
4. Add a meaningful deterministic ID/path through `_emit_catalog()`.
5. Ensure the source table name is correct so source validation and captions work.
6. Supply meaningful axis labels; they are persisted and checked in SVG metadata.
7. Avoid silent visual sampling. If necessary, use stable IDs/seeds and record the
   rendering policy.
8. Add a synthetic test for SVG parsing, style invariants, manifest row, and
   relevant semantics (for temporal plots, fixed limits/identity).
9. Update `visualization.md` and report section text when introducing a new topic.

## Test and verification commands

```bash
uv run --frozen python -m unittest -v \
  lab.ner.explicability.tests.test_part1 \
  lab.ner.explicability.tests.test_part2 \
  lab.ner.explicability.tests.test_part3
uv run --frozen python verification/verify_layering.py
uv run --frozen python -m compileall -q src verification
git diff --check
```

Trainer/model tests require the project's NER/PyTorch extra and local model
fixtures/cache. Documentation-only changes should still run the dependency-light
explicability and layering checks.
