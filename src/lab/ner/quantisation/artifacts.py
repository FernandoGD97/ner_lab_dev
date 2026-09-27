"""Safe artifact directories, hashing, and provenance files."""
from __future__ import annotations
import hashlib, importlib.metadata, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import yaml
from .model import checkpoint_size, inspect_checkpoint

MANIFEST = 'compression_manifest.json'; RECIPE = 'compression_recipe.yaml'
def source_hash(source):
    p=Path(source); h=hashlib.sha256()
    if p.exists():
        for f in sorted(x for x in p.rglob('*') if x.is_file()):
            h.update(str(f.relative_to(p)).encode()); h.update(f.read_bytes())
    else: h.update(str(source).encode())
    return h.hexdigest()
def prepare_output(source, output):
    source=Path(source).resolve() if Path(source).exists() else None; output=Path(output).resolve()
    if source is not None and output == source: raise ValueError('A compression artifact may not overwrite its source checkpoint.')
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'Output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    if source is not None:
        encoding = source / "encoding.json"
        if encoding.exists():
            shutil.copy2(encoding, output / encoding.name)
    return output
def write_artifact_metadata(source, output, method, metadata, backend='transformers', details=None,
                            parameter_count=None):
    info=inspect_checkpoint(output) if backend == 'transformers' else inspect_checkpoint(source)
    out=Path(output); source_bytes=checkpoint_size(source); result_bytes=checkpoint_size(out)
    versions = {}
    for package in ("transformers", "torch"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    detail_values = details or {}
    result_parameters = parameter_count if parameter_count is not None else info.parameter_count
    manifest={"schema_version":2,"artifact_id":source_hash(out)[:16],
      "source_model":str(source),"source_hash":source_hash(source),"task":detail_values.get("task", info.task),
      "architecture":info.architecture,"created_at":datetime.now(timezone.utc).isoformat(),
      "compression_method":method,"method_chain":[method],"dtype":info.dtype,
      "representation":detail_values.get("representation", method),
      "parameter_count":result_parameters,"original_parameter_count":info.parameter_count,
      "result_parameter_count":result_parameters,"checkpoint_bytes":result_bytes,
      "original_checkpoint_bytes":source_bytes,"result_checkpoint_bytes":result_bytes,
      "vocabulary_size":info.vocabulary_size,"hidden_size":info.hidden_size,
      "ffn_size":info.intermediate_size,"layers":info.number_of_layers,"heads":info.attention_heads,"backend":backend,
      "supported_devices":detail_values.get("supported_devices", ["cpu", "cuda"] if backend == "transformers" else ["cpu"]),
      "runtime_requirements":detail_values.get("runtime_requirements", {"backend": backend}),
      "seed":detail_values.get("seed"),"software_versions":versions,
      "scientific_references":[p.url for p in metadata.publications],"details":detail_values}
    (out/MANIFEST).write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (out/RECIPE).write_text(yaml.safe_dump({"model":str(source),"method":{"name":method},"output":{"path":str(out)}}))
    return manifest
