"""Keep the quantisation manual's command inventory aligned with the central parser."""
from pathlib import Path

from lab.cli import build_parser

parser = build_parser()
commands = [
    ["quantisation", "inspect", "MODEL"],
    ["quantisation", "methods"],
    ["quantisation", "method", "fp16"],
    ["quantisation", "fp16", "MODEL", "--output", "OUT"],
    ["quantisation", "bf16", "MODEL", "--output", "OUT"],
    ["quantisation", "int8", "MODEL", "--output", "OUT"],
    ["quantisation", "export-onnx", "MODEL", "--output", "OUT"],
    ["quantisation", "prune", "MODEL", "--output", "OUT"],
    ["quantisation", "prune-vocabulary", "MODEL", "--output", "OUT"],
    ["quantisation", "reduce-depth", "MODEL", "--output", "OUT", "--layers", "6"],
    ["quantisation", "svd", "MODEL", "--output", "OUT"],
    ["quantisation", "validate", "MODEL"],
    ["quantisation", "run-recipe", "recipe.yaml"],
    ["quantisation", "experiment", "validate", "study.yaml"],
    ["quantisation", "experiment", "run", "study.yaml", "--dry-run"],
    ["quantisation", "experiment", "status", "results"],
    ["quantisation", "experiment", "summarize", "results"],
    ["quantisation", "experiment", "compare", "results"],
    ["quantisation", "experiment", "analyse", "results"],
    ["quantisation", "experiment", "pareto", "results"],
    ["quantisation", "experiment", "correlations", "results"],
    ["quantisation", "experiment", "compare-predictions", "results", "--model", "A3"],
    ["quantisation", "experiment", "joint-report", "results", "--el-results", "el-results"],
    ["quantisation", "el", "validate", "el.yaml"],
    ["quantisation", "el", "run", "el.yaml"],
    ["quantisation", "el", "summarize", "el-results"],
]
for command in commands:
    arguments = parser.parse_args(command)
    assert callable(arguments.handler), command

manual = Path("docs/ner/quantisation/cli.md").read_text(encoding="utf-8")
for command in commands:
    phrase = " ".join(["lab", *command[:2]])
    assert phrase in manual, f"Missing documented command family: {phrase}"

assert "lab quantisation" in Path(
    "docs/ner/quantisation/CLI_CHEATSHEET.md"
).read_text(encoding="utf-8")
