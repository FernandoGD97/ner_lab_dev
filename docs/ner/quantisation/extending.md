# Developer guide: extending quantisation

[Home](README.md) · [Architecture](system-architecture.md) · [Tests](../../../verification/README.md)

## Add a compression method

1. Choose/create a family module under `src/lab/ner/quantisation/<family>/`.
2. Define `MethodMetadata` with a unique registry name, family, one of the three
   `CompressionCategory` values, `TrainingRequirement`, support labels, `Fidelity`, limitations,
   `Publication` entries, and implementation URLs.
3. Subclass `CompressionMethod`. Override `check_compatibility` if callers using `__call__` need
   preflight errors, and implement `apply(source, output, **kwargs)`.
4. Start `apply` with `prepare_output`; never overwrite the source. Load through `load_checkpoint`
   when compatible, save model/tokenizer/runtime files, and call `write_artifact_metadata`.
5. Use a truthful backend. Normal AutoModel-loadable output uses `transformers`; specialized output
   needs its loader and corresponding branch in validation/canonical loading before it is usable.
6. Call `self.validate(out)` for normal artifacts where practical. Do not claim validation for a
   runtime the validator cannot execute.
7. Add the method instance to `_METHODS` in `registry.py`. Add bibliography context separately to
   `references/bibliography.yaml`; there is no automatic bibliography generator.
8. Add a thin handler/Typer command and central `add_parser` entry only if a stable user operation
   exists. Metadata-only scaffolds should not acquire misleading commands.
9. Add CPU-small verification, help/registry checks, a method page, limitations, and a tutorial if
   appropriate.

Minimal skeleton (illustrative; choose real metadata and checks):

```python
from ..artifacts import prepare_output, write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import CompressionCategory, Fidelity, MethodMetadata, Publication, TrainingRequirement
from ..model import load_checkpoint

META = MethodMetadata(
    name="my-method",
    family="my-family",
    category=CompressionCategory.MODEL_COMPRESSION,
    training_requirement=TrainingRequirement.NONE,
    fidelity=Fidelity.GENERIC_EQUIVALENT,
    limitations=("State the real limitation.",),
    publications=(Publication("key", "Title", "https://..."),),
)

class MyMethod(CompressionMethod):
    metadata = META

    def apply(self, source, output, task="auto", **kwargs):
        out = prepare_output(source, output)
        model, tokenizer, _ = load_checkpoint(source, task)
        # Transform model truthfully.
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        write_artifact_metadata(source, out, self.metadata.name, self.metadata)
        self.validate(out)
        return out
```

Important current nuance: CLI `apply_command` calls `apply` directly, not the base `__call__`; a
compatibility check only in `check_compatibility` is therefore not enforced by existing CLI commands.
Put safety-critical checks in `apply` unless command plumbing is deliberately changed and tested.

## Add a Typer command

The Typer apps live in `lab.ner.quantisation.cli`:

```python
@app.command("my-command")
def typer_my_command(model: str, output: Path) -> None:
    apply_command("my-method", model, output)
```

For a nested command, decorate with `@experiment_app.command(...)` or `@el_app.command(...)`.
Required parameters without defaults become Typer arguments; defaults become options. This codebase
uses `typer.Option(...)` only when an option must be explicitly required.

Because the installed executable is the central argparse CLI, adding only the decorator is
insufficient. Add a matching parser inside `add_parser` and route it to the same ordinary handler.
Then run both central `--help` and `typer.testing.CliRunner`. Do not put method logic in either CLI
wrapper.

Pydantic validation is invoked explicitly inside handlers (`load_recipe`, `load_experiment`, or
`load_el_experiment`). `_echo` renders JSON with `default=str`. The central `lab.cli.main` converts
`ValueError`, `FileNotFoundError`, and `TypeError` to exit 2; other exceptions are not normalized.

## Add a runtime/backend

1. Define precisely whether the artifact is normal Hugging Face or specialized.
2. Detect optional modules inside functions, not at import time; never wrap imports in try/catch at
   module top level. `quantization/backends.py` demonstrates `importlib.util.find_spec` reporting.
3. Save backend identity and provider-relevant details in the manifest.
4. Add a loader and validation branch. If normal NER inference can support it without redesign,
   add manifest-aware loading; otherwise mark it unsupported in `experiment/matrix.py`.
5. Add timing boundaries that separate compilation/initialization from repeated execution and store
   runtime/provider as explicit factors.
6. Do not compare unlike runtime families as if precision were the only treatment.
7. Test missing-dependency errors, CPU paths, reload, finite output, schema, and help. Document how
   to install and select providers.

ONNX currently stops at export; it is a good example of a backend that must remain explicitly
unsupported in the PyTorch benchmark until a real adapter exists.

## Add configuration

Add fields to the relevant Pydantic model with constraints/defaults, decide whether unknown fields
are forbidden, ensure `model_dump(mode="json")` can write the resolved YAML, and document exact
precedence. Avoid accepting fields that orchestration ignores.

## Add tests

This repository uses executable checks under `verification/`, not a `tests/` pytest tree. Add small,
local CPU fixtures under `verification/quantisation/`, assert both Typer and central CLI wiring, and
extend `verify_layering.py` expectations by preserving lazy cross-namespace imports. GPU,
CodeCarbon, FAISS, and network checks must skip or be separated from normal checks.
