# Tutorial 1 — inspect a model

[Tutorials](README.md) · [CLI discovery](../cli.md#discovery-and-inspection-commands)

```bash
MODEL=models/biomedical-ner
lab quantisation inspect "$MODEL" --task token-classification > inspection.json
lab quantisation methods > methods.json
lab quantisation method fp16
lab quantisation method int8-dynamic
```

Confirm the loaded class/task, label count, vocabulary, hidden/FFN sizes, layers/heads, and dtype.
Read method limitations and fidelity. Metadata support lists are not a live compatibility check;
transformation plus `validate --source` is the actual technical test. If the model is a normal
fine-tuned NER artifact, confirm it has `encoding.json` before planning a controlled experiment.
