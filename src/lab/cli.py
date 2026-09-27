"""Command-line entry point: `lab run <config.yaml>` and `lab tasks`."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Sequence

import yaml

from lab import __version__
from lab.core.tasks import list_tasks, resolve_task

OVERRIDES = ("random_state", "output_dir")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(config).__name__}."
        )

    return config


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    name = config.get("task")

    if name is None:
        raise ValueError("Config has no 'task' key. `lab tasks` lists the available ones.")

    task = resolve_task(name)
    parameters = {key: value for key, value in config.items() if key != "task"}

    for override in OVERRIDES:
        value = getattr(args, override)

        if value is not None:
            parameters[override] = value

    task(**parameters)


def tasks(args: argparse.Namespace) -> None:
    listed = list_tasks()
    width = max(len(name) for name, _ in listed)

    for name, missing in listed:
        namespace = name.partition(".")[0]
        note = f"  needs: pip install lab[{namespace}]" if missing else ""

        print(f"{name:<{width}}{note}")


def finalize_explicability(args: argparse.Namespace) -> None:
    finalize_run = importlib.import_module("lab.ner.explicability").finalize_run
    finalize_run(
        args.run_directory,
        completed_analyses=args.completed_analysis,
        required_outputs=args.required_output,
        mandatory_analyses=args.mandatory_analysis,
    )


def analyze_explicability(args: argparse.Namespace) -> None:
    run_analyses = importlib.import_module(
        "lab.ner.explicability.analyses"
    ).run_analyses
    run_analyses(args.run_directory)


def plot_explicability(args: argparse.Namespace) -> None:
    generate_figures = importlib.import_module(
        "lab.ner.explicability.visualization"
    ).generate_figures
    generate_figures(
        args.run_directory,
        profile_name=args.profile,
        analysis=args.analysis,
        figure_id=args.figure,
        animations=not args.no_animation,
    )


def animate_explicability(args: argparse.Namespace) -> None:
    regenerate = importlib.import_module(
        "lab.ner.explicability.visualization.animations"
    ).regenerate_animation
    regenerate(args.run_directory)


def compare_explicability(args: argparse.Namespace) -> None:
    module = importlib.import_module("lab.ner.explicability.visualization.comparison")
    config_type = importlib.import_module("lab.ner.explicability.config").VisualizationConfig
    module.compare_runs(args.run_directories, args.output_directory,
                        config_type(profile=args.profile))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab",
        description="Run one of the library's tasks from a YAML config.",
    )
    parser.add_argument("--version", action="version", version=f"lab {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    run_parser = subparsers.add_parser("run", help="Run the task described by a YAML config.")
    run_parser.add_argument("config", help="Path to the YAML config.")
    run_parser.add_argument(
        "--random-state",
        "--seed",
        dest="random_state",
        type=int,
        default=None,
        help="Override the config's random_state.",
    )
    run_parser.add_argument("--output-dir", default=None, help="Override the config's output_dir.")
    run_parser.set_defaults(handler=run)

    tasks_parser = subparsers.add_parser(
        "tasks", help="List every task, marking those whose extra is not installed."
    )
    tasks_parser.set_defaults(handler=tasks)

    # Imported through importlib so the central CLI keeps its namespace layering:
    # compression dependencies are resolved only while constructing this command group.
    import importlib

    importlib.import_module("lab.ner.quantisation.cli").add_parser(subparsers)
    explicability_parser = subparsers.add_parser(
        "explicability", help="Manage representation-analysis artifacts."
    )
    explicability_subparsers = explicability_parser.add_subparsers(
        dest="explicability_command", metavar="<command>"
    )
    finalize_parser = explicability_subparsers.add_parser(
        "finalize", help="Validate and finalize an already-trained run."
    )
    finalize_parser.add_argument("run_directory")
    finalize_parser.add_argument("--completed-analysis", action="append", default=[])
    finalize_parser.add_argument("--mandatory-analysis", action="append", default=None)
    finalize_parser.add_argument("--required-output", action="append", default=None)
    finalize_parser.set_defaults(handler=finalize_explicability)

    analyze_parser = explicability_subparsers.add_parser(
        "analyze", help="Transform snapshots into Part 2 Parquet analyses."
    )
    analyze_parser.add_argument("run_directory")
    analyze_parser.set_defaults(handler=analyze_explicability)

    plot_parser = explicability_subparsers.add_parser(
        "plot", help="Regenerate Part 3 SVG figures and report from Parquet."
    )
    plot_parser.add_argument("run_directory")
    plot_parser.add_argument("--profile", choices=("paper", "presentation"), default=None)
    plot_parser.add_argument("--analysis", default=None)
    plot_parser.add_argument("--figure", default=None)
    plot_parser.add_argument("--no-animation", action="store_true")
    plot_parser.set_defaults(handler=plot_explicability)

    animate_parser = explicability_subparsers.add_parser(
        "animate", help="Regenerate temporal animation from projection Parquet."
    )
    animate_parser.add_argument("run_directory")
    animate_parser.set_defaults(handler=animate_explicability)

    compare_parser = explicability_subparsers.add_parser(
        "compare", help="Compare completed runs from permanent fingerprint tables."
    )
    compare_parser.add_argument("output_directory")
    compare_parser.add_argument("run_directories", nargs="+")
    compare_parser.add_argument("--profile", choices=("paper", "presentation"), default="paper")
    compare_parser.set_defaults(handler=compare_explicability)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 1

    try:
        args.handler(args)
    except (ValueError, FileNotFoundError, TypeError) as error:
        parser.exit(2, f"lab: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
