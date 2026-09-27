"""Safe artifact directories, hashing, and provenance files."""
from __future__ import annotations
import hashlib, importlib.metadata, json, shutil
from pathlib import Path
import yaml
from .model import inspect_checkpoint

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
    manifest={"schema_version":1,"source_model":str(source),"source_hash":source_hash(source),
      "compression_method":method,"method_chain":[method],"dtype":info.dtype,"parameter_count":parameter_count if parameter_count is not None else info.parameter_count,
      "checkpoint_bytes":info.checkpoint_bytes,"vocabulary_size":info.vocabulary_size,"hidden_size":info.hidden_size,
      "ffn_size":info.intermediate_size,"layers":info.number_of_layers,"heads":info.attention_heads,"backend":backend,
      "software_versions":{x:importlib.metadata.version(x) for x in ('transformers','torch')},
      "scientific_references":[p.url for p in metadata.publications],"details":details or {}}
    out=Path(output); (out/MANIFEST).write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (out/RECIPE).write_text(yaml.safe_dump({"model":str(source),"method":{"name":method},"output":{"path":str(out)}}))
    return manifest
