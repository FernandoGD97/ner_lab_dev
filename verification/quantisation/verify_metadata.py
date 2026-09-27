"""Dependency-light checks for the compression registry and bibliography."""
from lab.ner.quantisation.registry import get_method, list_methods
from lab.ner.quantisation.references import load_references

methods = list_methods()
assert {"fp32", "fp16", "bf16", "int8-dynamic", "vocabulary-pruning"} <= {
    method.name for method in methods
}
for method in methods:
    assert method.publications
    assert all(publication.url.startswith("https://") for publication in method.publications)
assert len(load_references()) >= 20
try:
    get_method("not-a-method")
except ValueError as error:
    assert "Unsupported compression method" in str(error)
else:
    raise AssertionError("unsupported methods must fail")
