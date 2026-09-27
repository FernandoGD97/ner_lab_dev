"""Guard the analysis guide against claiming a nonexistent CLI or stale schema."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from _harness import Checks, run
from lab.ner.analysis import __all__ as public_api
from lab.ner.analysis.parquet import evaluation_schema


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "src" / "lab" / "ner" / "analysis" / "README.md"


def main() -> int:
    checks = Checks("analysis documentation")
    text = GUIDE.read_text(encoding="utf-8")

    checks.check("the package-local researcher guide exists", GUIDE.is_file())
    checks.equal(
        "there is no analysis module entry point to document",
        importlib.util.find_spec("lab.ner.analysis.__main__"),
        None,
    )
    checks.check("the guide explicitly states that no Typer CLI exists", "no Typer CLI" in text)
    checks.check(
        "the guide does not present a fake module command as runnable",
        "```bash\npython -m lab.ner.analysis" not in text,
    )

    for name in (
        "analyze_evaluation",
        "load_evaluation",
        "create_publication_report",
        "oracle_error_budget",
        "bootstrap_intervals",
        "paired_seen_zero_shot_difference",
        "AnalysisConfig",
        "NormalizationConfig",
        "BootstrapConfig",
        "PublicationConfig",
    ):
        checks.check(f"public API {name} is documented", name in public_api and f"`{name}" in text)

    missing_columns = [field.name for field in evaluation_schema() if f"`{field.name}`" not in text]
    checks.equal("every current Parquet field is named in the data dictionary", missing_columns, [])

    for filename in (
        "evaluation.parquet",
        "oracle_error_budget.parquet",
        "bootstrap_intervals.parquet",
        "generalization_strict_f1.svg",
        "oracle_error_budget.svg",
        "captions.md",
        "findings.md",
        "report_manifest.json",
    ):
        checks.check(f"output {filename} is documented", filename in text)

    checks.check("the guide documents the 0.80 threshold", "`0.80`" in text)
    checks.check("the guide documents non-additive oracle gains", "not additive" in text)
    checks.check("the guide includes developer extension guidance", "## Developer extension guide" in text)

    return checks.report()


if __name__ == "__main__":
    run(main)
