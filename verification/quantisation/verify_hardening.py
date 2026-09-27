"""Offline checks for executable compression registration and deterministic contracts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml
from pydantic import ValidationError
from lab.cli import build_parser
from lab.ner.quantisation.config import load_recipe
from lab.ner.quantisation.distillation.mapping import evenly_spaced_layer_mapping
from lab.ner.quantisation.registry import get_method

assert evenly_spaced_layer_mapping(12, 6) == (0, 2, 4, 7, 9, 11)
assert evenly_spaced_layer_mapping(12, 1) == (11,)
for method in ('int8-static','int4','structured-pruning','student'):
    assert get_method(method).metadata.name == method
parser=build_parser()
for arguments in (
 ('quantisation','int4','model','--output','out','--group-size','32'),
 ('quantization','int8-static','model','--output','out','--calibration-corpus','calibration.txt'),
 ('quantisation','prune-structured','model','--output','out','--head-amount','0.25'),
 ('quantisation','student','model','--output','out','--layers','6','--hidden-size','384'),
):
    assert parser.parse_args(arguments).q_handler
with TemporaryDirectory() as root:
    path=Path(root)/'recipe.yaml';path.write_text(yaml.safe_dump({'model':'m','method':{'name':'int4','unknown':1},'output':{'path':'o'}}))
    try: load_recipe(path)
    except ValidationError: pass
    else: raise AssertionError('Unknown recipe fields must be rejected')
print('quantisation hardening checks passed')
