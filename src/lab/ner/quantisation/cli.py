"""Compression and controlled-experiment commands for the central CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import typer

from .model import inspect_checkpoint
from .registry import get_method, list_methods, method_metadata
from .validation import validate_checkpoint

app = typer.Typer(help="Inspect and compress pretrained Transformer checkpoints.", no_args_is_help=True)
experiment_app = typer.Typer(help="Run controlled compression benchmarks.", no_args_is_help=True)
el_app = typer.Typer(help="Benchmark the existing entity-linking pipeline.", no_args_is_help=True)
app.add_typer(experiment_app, name="experiment")
app.add_typer(el_app, name="el")


def _echo(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def inspect_command(model: str, task: str = "auto") -> None:
    _echo(inspect_checkpoint(model, task).to_dict())


def methods_command() -> None:
    _echo([metadata.to_dict() for metadata in list_methods()])


def method_command(name: str) -> None:
    _echo(method_metadata(name).to_dict())


def apply_command(name: str, model: str, output: Path, **kwargs: Any) -> None:
    values = {key: value for key, value in kwargs.items() if value is not None}
    print(get_method(name).apply(model, output, **values))


def validate_command(model: str, source: str | None = None) -> None:
    _echo(validate_checkpoint(model, source).to_dict())


def recipe_command(recipe: Path, output: Path | None = None) -> None:
    from .config import load_recipe

    config = load_recipe(recipe)
    values = config.method.model_dump(exclude={"name", "mode"}, exclude_none=True)
    apply_command(config.method.name, config.model, output or config.output.path, **values)


def experiment_validate(config: Path) -> None:
    from .experiment.runner import validate_experiment

    _echo(validate_experiment(config))


def experiment_run(
    config: Path,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
) -> None:
    from .experiment.runner import run_experiment

    _echo(run_experiment(config, dry_run=dry_run, resume=resume, force=force))


def experiment_status_command(results_dir: Path) -> None:
    from .experiment.runner import experiment_status

    _echo(experiment_status(results_dir))


def experiment_summarize(results_dir: Path) -> None:
    from .experiment.results import collect_runs, write_summary

    print(write_summary(collect_runs(results_dir), results_dir / "benchmark_summary.tsv"))


def experiment_compare(results_dir: Path, baseline: str = "A0") -> None:
    from .experiment.results import collect_runs, write_comparison

    print(write_comparison(collect_runs(results_dir), results_dir / "comparison.tsv", baseline))


def experiment_analyse(results_dir: Path, training_frequency_source: Path | None = None,
                       baseline: str = "A0", plots: bool = False) -> None:
    from .analysis.workflow import analyse_results
    _echo(analyse_results(results_dir, training_frequency_source, baseline, plots=plots))


def experiment_pareto(results_dir: Path) -> None:
    import pandas as pd
    from .analysis.pareto import export_pareto
    from .analysis.reporting import model_level
    frame = pd.read_csv(results_dir / "benchmark_long.tsv", sep="\t")
    _echo({key: str(path) for key, path in export_pareto(model_level(frame), results_dir / "analysis").items()})


def experiment_correlations(results_dir: Path) -> None:
    import pandas as pd
    from .analysis.reporting import correlations
    output = results_dir / "analysis" / "correlations.tsv"; output.parent.mkdir(parents=True, exist_ok=True)
    correlations(pd.read_csv(results_dir / "benchmark_long.tsv", sep="\t")).to_csv(output, sep="\t", index=False)
    print(output)


def experiment_compare_predictions(results_dir: Path, baseline: str, model: str,
                                   training_frequency_source: Path | None = None) -> None:
    from .analysis.workflow import compare_stored_predictions
    _echo(compare_stored_predictions(results_dir, baseline, model, training_frequency_source))


def experiment_joint_report(results_dir: Path, el_results: Path, baseline: str = "A0") -> None:
    import pandas as pd
    from .analysis.reporting import joint_task_summary
    output = results_dir / "analysis" / "joint_ner_el.tsv"; output.parent.mkdir(parents=True, exist_ok=True)
    joint_task_summary(
        pd.read_csv(results_dir / "benchmark_long.tsv", sep="\t"),
        pd.read_csv(el_results / "el_benchmark_long.tsv", sep="\t"), baseline,
    ).to_csv(output, sep="\t", index=False)
    print(output)


def el_validate(config: Path) -> None:
    from .entity_linking import validate_el
    _echo(validate_el(config))


def el_run(config: Path) -> None:
    from .entity_linking import run_el_experiment
    _echo(run_el_experiment(config))


def el_summarize(results_dir: Path) -> None:
    from .entity_linking import summarize_el
    print(summarize_el(results_dir))


@app.command("inspect")
def typer_inspect(model: str, task: str = "auto") -> None:
    inspect_command(model, task)


@app.command("methods")
def typer_methods() -> None:
    methods_command()


@app.command("method")
def typer_method(name: str) -> None:
    method_command(name)


@app.command("fp16")
def typer_fp16(model: str, output: Path, task: str = "auto") -> None:
    apply_command("fp16", model, output, task=task)


@app.command("bf16")
def typer_bf16(model: str, output: Path, task: str = "auto") -> None:
    apply_command("bf16", model, output, task=task)


@app.command("int8")
def typer_int8(model: str, output: Path, task: str = "auto") -> None:
    apply_command("int8-dynamic", model, output, task=task)


@app.command("prune")
def typer_prune(model: str, output: Path, amount: float = 0.2, task: str = "auto") -> None:
    apply_command("magnitude-pruning", model, output, amount=amount, task=task)


@app.command("prune-vocabulary")
def typer_vocab(
    model: str, output: Path, keep_token: list[str] | None = None, task: str = "auto"
) -> None:
    apply_command("vocabulary-pruning", model, output, keep_tokens=keep_token or [], task=task)


@app.command("reduce-depth")
def typer_depth(model: str, output: Path, layers: int, task: str = "auto") -> None:
    apply_command("depth-reduction", model, output, layers=layers, task=task)


@app.command("svd")
def typer_svd(model: str, output: Path, rank_ratio: float = 0.5, task: str = "auto") -> None:
    apply_command("svd", model, output, rank_ratio=rank_ratio, task=task)


@app.command("export-onnx")
def typer_onnx(model: str, output: Path, task: str = "auto") -> None:
    apply_command("onnx", model, output, task=task)


@app.command("validate")
def typer_validate(model: str, source: str | None = None) -> None:
    validate_command(model, source)


@app.command("run-recipe")
def typer_recipe(recipe: Path, output: Path | None = None) -> None:
    recipe_command(recipe, output)


@experiment_app.command("validate")
def typer_experiment_validate(config: Path) -> None:
    experiment_validate(config)


@experiment_app.command("run")
def typer_experiment_run(
    config: Path,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
) -> None:
    experiment_run(config, dry_run, resume, force)


@experiment_app.command("status")
def typer_experiment_status(results_dir: Path) -> None:
    experiment_status_command(results_dir)


@experiment_app.command("summarize")
def typer_experiment_summarize(results_dir: Path) -> None:
    experiment_summarize(results_dir)


@experiment_app.command("compare")
def typer_experiment_compare(results_dir: Path, baseline: str = "A0") -> None:
    experiment_compare(results_dir, baseline)


@experiment_app.command("analyse")
def typer_experiment_analyse(results_dir: Path, training_frequency_source: Path | None = None,
                             baseline: str = "A0", plots: bool = False) -> None:
    experiment_analyse(results_dir, training_frequency_source, baseline, plots)


@experiment_app.command("pareto")
def typer_experiment_pareto(results_dir: Path) -> None:
    experiment_pareto(results_dir)


@experiment_app.command("correlations")
def typer_experiment_correlations(results_dir: Path) -> None:
    experiment_correlations(results_dir)


@experiment_app.command("compare-predictions")
def typer_experiment_compare_predictions(results_dir: Path, baseline: str = "A0",
                                         model: str = typer.Option(...),
                                         training_frequency_source: Path | None = None) -> None:
    experiment_compare_predictions(results_dir, baseline, model, training_frequency_source)


@experiment_app.command("joint-report")
def typer_experiment_joint_report(results_dir: Path, el_results: Path,
                                  baseline: str = "A0") -> None:
    experiment_joint_report(results_dir, el_results, baseline)


@el_app.command("validate")
def typer_el_validate(config: Path) -> None:
    el_validate(config)


@el_app.command("run")
def typer_el_run(config: Path) -> None:
    el_run(config)


@el_app.command("summarize")
def typer_el_summarize(results_dir: Path) -> None:
    el_summarize(results_dir)


def _dispatch(args: argparse.Namespace) -> None:
    values = vars(args).copy()
    action = values.pop("q_handler")
    for key in ("handler", "command", "quant_command", "experiment_command", "el_command"):
        values.pop(key, None)
    action(**values)


def _command(subparsers, name: str, handler):
    parser = subparsers.add_parser(name)
    parser.set_defaults(handler=_dispatch, q_handler=handler)
    return parser


def _transformation_parser(commands, command: str, method: str):
    parser = _command(
        commands,
        command,
        lambda model, output, task="auto", selected=method: apply_command(
            selected, model, output, task=task
        ),
    )
    parser.add_argument("model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", default="auto")


def add_parser(subparsers) -> None:
    quantisation = subparsers.add_parser(
        "quantisation",
        aliases=["quantization"],
        help="Inspect, compress, and benchmark Transformer checkpoints.",
    )
    commands = quantisation.add_subparsers(dest="quant_command", required=True)

    parser = _command(commands, "inspect", inspect_command)
    parser.add_argument("model")
    parser.add_argument("--task", default="auto")
    _command(commands, "methods", methods_command)
    parser = _command(commands, "method", method_command)
    parser.add_argument("name")
    for command, method in (
        ("fp16", "fp16"), ("bf16", "bf16"), ("int8", "int8-dynamic"),
        ("export-onnx", "onnx"),
    ):
        _transformation_parser(commands, command, method)

    parser = _command(
        commands, "prune",
        lambda model, output, amount=0.2, task="auto": apply_command(
            "magnitude-pruning", model, output, amount=amount, task=task
        ),
    )
    parser.add_argument("model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--amount", type=float, default=0.2)
    parser.add_argument("--task", default="auto")

    parser = _command(
        commands, "prune-vocabulary",
        lambda model, output, keep_token, task="auto": apply_command(
            "vocabulary-pruning", model, output, keep_tokens=keep_token, task=task
        ),
    )
    parser.add_argument("model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keep-token", action="append", default=[])
    parser.add_argument("--task", default="auto")

    parser = _command(
        commands, "reduce-depth",
        lambda model, output, layers, task="auto": apply_command(
            "depth-reduction", model, output, layers=layers, task=task
        ),
    )
    parser.add_argument("model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=int)
    parser.add_argument("--task", default="auto")

    parser = _command(
        commands, "svd",
        lambda model, output, rank_ratio=0.5, task="auto": apply_command(
            "svd", model, output, rank_ratio=rank_ratio, task=task
        ),
    )
    parser.add_argument("model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank-ratio", type=float, default=0.5)
    parser.add_argument("--task", default="auto")

    parser = _command(commands, "validate", validate_command)
    parser.add_argument("model")
    parser.add_argument("--source")
    parser = _command(commands, "run-recipe", recipe_command)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--output", type=Path)

    experiment = commands.add_parser("experiment", help="Controlled benchmark experiments.")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    parser = _command(experiment_commands, "validate", experiment_validate)
    parser.add_argument("config", type=Path)
    parser = _command(experiment_commands, "run", experiment_run)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser = _command(experiment_commands, "status", experiment_status_command)
    parser.add_argument("results_dir", type=Path)
    parser = _command(experiment_commands, "summarize", experiment_summarize)
    parser.add_argument("results_dir", type=Path)
    parser = _command(experiment_commands, "compare", experiment_compare)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--baseline", default="A0")
    parser = _command(experiment_commands, "analyse", experiment_analyse)
    parser.add_argument("results_dir", type=Path); parser.add_argument("--training-frequency-source", type=Path)
    parser.add_argument("--baseline", default="A0"); parser.add_argument("--plots", action="store_true")
    parser = _command(experiment_commands, "pareto", experiment_pareto); parser.add_argument("results_dir", type=Path)
    parser = _command(experiment_commands, "correlations", experiment_correlations); parser.add_argument("results_dir", type=Path)
    parser = _command(experiment_commands, "compare-predictions", experiment_compare_predictions)
    parser.add_argument("results_dir", type=Path); parser.add_argument("--baseline", default="A0")
    parser.add_argument("--model", required=True); parser.add_argument("--training-frequency-source", type=Path)
    parser = _command(experiment_commands, "joint-report", experiment_joint_report)
    parser.add_argument("results_dir", type=Path); parser.add_argument("--el-results", required=True, type=Path)
    parser.add_argument("--baseline", default="A0")

    el = commands.add_parser("el", help="Entity-linking component benchmarks.")
    el_commands = el.add_subparsers(dest="el_command", required=True)
    parser = _command(el_commands, "validate", el_validate); parser.add_argument("config", type=Path)
    parser = _command(el_commands, "run", el_run); parser.add_argument("config", type=Path)
    parser = _command(el_commands, "summarize", el_summarize); parser.add_argument("results_dir", type=Path)
