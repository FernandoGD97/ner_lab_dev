"""Self-contained scientific HTML report assembled from permanent artifacts."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


SECTION_TEXT = {
    "representations": ("Representation adaptation", "Shared coordinates show how stable observations reorganize. Apparent movement is interpreted only in the common projection; projection geometry is not causal. See McInnes et al. (2018)."),
    "trajectories": ("Representation trajectories", "Path length and displacement quantify movement through training. Large movement is descriptive and does not by itself imply better NER behavior."),
    "geometry": ("Representation geometry", "Compactness, separation, intrinsic dimension, and anisotropy describe organization at each checkpoint. Small groups should be interpreted cautiously."),
    "dynamics": ("Example learning dynamics", "Confidence variability, forgetting, and label transitions describe when observations are learned or destabilized. See Swayamdipta et al. (2020) and Toneva et al. (2019)."),
    "neighborhoods": ("Neighborhood evolution", "Top-k overlap and purity show local semantic reorganization without retaining full distance matrices."),
    "probing": ("Layer-wise probing", "Frozen probes measure decodability, not causal use of information by the NER model. See Alain and Bengio (2017)."),
    "parameters": ("Parameter adaptation", "Relative weight changes locate where optimization modified the architecture; they do not identify causal mechanisms."),
    "composites": ("Run summary", "Aligned panels place performance and internal diagnostics on separate scales to avoid misleading dual axes."),
}


def draft_caption(record: dict) -> str:
    population = record.get("representation_level") or "available observations"
    checkpoints = record.get("checkpoints") or "recorded checkpoints"
    return (
        f"{record.get('analysis', 'Analysis')} for {population} across {checkpoints}. "
        f"Values are derived from {record.get('source_table', 'the permanent analytical table')}; "
        "visual encodings do not imply causality."
    )


def generate_report(root: str | Path, figure_manifest: pd.DataFrame, run_manifest: dict) -> Path:
    root = Path(root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    for category, group in figure_manifest.groupby("category", sort=False):
        heading, interpretation = SECTION_TEXT.get(category, (str(category).title(), "This section summarizes permanent analytical outputs without adding causal claims."))
        figures = []
        for row in group.to_dict("records"):
            relative = Path("..") / Path(row["filename"])
            figures.append(
                f'<figure id="{html.escape(str(row["figure_id"]))}">'
                f'<object data="{html.escape(str(relative))}" type="image/svg+xml" loading="lazy"></object>'
                f'<figcaption>{html.escape(draft_caption(row))}</figcaption></figure>'
            )
        sections.append(f'<section><h2>{html.escape(heading)}</h2><p>{html.escape(interpretation)}</p>{"".join(figures)}</section>')
    animations = []
    for path in sorted((root / "animations").glob("*")):
        relative = Path("..") / path.relative_to(root)
        animations.append(f'<li><a href="{html.escape(str(relative))}">{html.escape(path.name)}</a></li>')
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>NER explicability report</title><style>
body{{font-family:Arial,Helvetica,sans-serif;color:#222;max-width:1200px;margin:auto;padding:2rem;line-height:1.5}}h1,h2{{font-weight:600}}.meta{{display:grid;grid-template-columns:max-content 1fr;gap:.3rem 1rem;background:#f5f5f5;padding:1rem}}figure{{margin:2rem 0}}object{{width:100%;min-height:360px;border:0}}figcaption{{font-size:.9rem;color:#555;max-width:85ch}}input{{padding:.55rem;width:min(28rem,90%);border:1px solid #999}}@media(max-width:700px){{body{{padding:1rem}}object{{min-height:260px}}}}</style></head><body>
<h1>NER corpus-adaptation report</h1><div class="meta"><b>Run</b><span>{html.escape(str(run_manifest.get("run_id")))}</span><b>Model</b><span>{html.escape(str(run_manifest.get("model")))}</span><b>Dataset</b><span>{html.escape(str(run_manifest.get("dataset")))}</span><b>Split</b><span>{html.escape(str(run_manifest.get("split")))}</span></div>
<p>This report separates measured quantities from interpretation. Correlations, probes, projections, attention, and attribution must not be read as causal evidence.</p><label>Filter figures: <input id="figure-filter" placeholder="e.g. CKA, forgetting, probing"></label>{''.join(sections)}<section><h2>Animations and supplementary exploration</h2><ul>{''.join(animations) or '<li>No animation was generated.</li>'}</ul></section><script>document.getElementById('figure-filter').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('figure').forEach(f=>f.hidden=!f.textContent.toLowerCase().includes(q)&&!f.id.toLowerCase().includes(q));}});</script></body></html>'''
    path = report_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path
