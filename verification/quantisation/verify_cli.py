"""The existing central CLI owns compression and experiment command groups."""
from lab.cli import build_parser

parser = build_parser()
args = parser.parse_args(["quantisation", "methods"])
assert args.quant_command == "methods"
assert callable(args.handler)
args = parser.parse_args(["quantisation", "experiment", "run", "study.yaml", "--dry-run"])
assert args.experiment_command == "run"
assert args.dry_run is True
assert callable(args.handler)

args = parser.parse_args(["quantisation", "experiment", "compare-predictions", "results", "--model", "A3"])
assert args.experiment_command == "compare-predictions"
args = parser.parse_args(["quantisation", "el", "validate", "el.yaml"])
assert args.el_command == "validate"
