from pathlib import Path
import yaml
def load_references():
 with (Path(__file__).with_name('bibliography.yaml')).open(encoding='utf8') as f:return yaml.safe_load(f)['references']
