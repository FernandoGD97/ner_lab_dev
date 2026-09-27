"""Recipe and controlled-experiment Pydantic validation checks."""
import tempfile
from pathlib import Path
from lab.ner.quantisation.config import load_recipe
from lab.ner.quantisation.experiment.config import load_experiment
from lab.ner.quantisation.entity_linking.config import load_el_experiment

recipes = Path("src/lab/ner/quantisation/recipes")
assert len(list(recipes.glob("*.yaml"))) == 22
for recipe in recipes.glob("*.yaml"):
    parsed = load_recipe(recipe)
    assert parsed.status in {"READY", "REQUIRES_TRAINING", "EXPERIMENTAL", "UNSUPPORTED"}

with tempfile.TemporaryDirectory() as temporary:
    config = Path(temporary) / "experiment.yaml"
    config.write_text("""
experiment: {id: test, seed: 7}
dataset: {path: test.parquet, split: test}
inference: {batch_size: 4, max_length: 128, warmup_batches: 2, repetitions: 3}
models:
  - {id: A0, path: a0, runtime: pytorch}
  - {id: A1, path: a1, runtime: pytorch}
energy: {enabled: false}
output: {directory: results}
protocol: controlled
""")
    experiment = load_experiment(config)
    assert experiment.experiment.seed == 7
    assert experiment.inference.repetitions == 3
    assert experiment.models[0].id == "A0"
    el_config = Path(temporary) / "el.yaml"
    el_config.write_text("""
experiment: {id: el_test, seed: 7}
task: {type: entity_linking, mentions: mentions.tsv, gazetteer: terms.tsv}
retrieval: {candidate_k: [200, 50, 10], method: matrix, index_type: flat_ip}
output: {directory: results}
""")
    el = load_el_experiment(el_config)
    assert el.retrieval.candidate_k == [200, 50, 10]
    assert el.task.type == "entity_linking"
