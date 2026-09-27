# Scientific references and fidelity

[Home](README.md) · [Method index](methods/README.md) · [Extending](extending.md)

## Two reference stores

1. Each registered method has `MethodMetadata.publications`, a tuple of immutable
   `Publication(key, title, url)` records. There is no `PaperReference` class.
2. `src/lab/ner/quantisation/references/bibliography.yaml` is a broader hand-maintained mapping
   loaded by `references.load_references()`. It includes named methods that are **not implemented**.

The two stores are not automatically synchronized and no bibliography generator exists. A method's
publication URLs appear in `lab quantisation methods`, `lab quantisation method NAME`, and generated
manifest `scientific_references`. Bibliography-only entries do not appear in a method manifest.

## Fidelity prevents false attribution

`EXACT`, `APPROXIMATION`, `PAPER_INSPIRED`, `GENERIC_EQUIVALENT`, and `EXTERNAL_ADAPTER` describe
implementation relationship, not paper quality. Generic INT8 is not Q-BERT/I-BERT; prefix depth
reduction is not LayerDrop training; generic losses are not TinyBERT/MiniLM.

## Bibliography scope

The YAML links Knowledge Distillation, DistilBERT, TinyBERT, MobileBERT, MiniLM, ALBERT, Q-BERT,
TernaryBERT, I-BERT, BinaryBERT, ZeroQuant, Movement/Block Pruning, CoFi, PoWER-BERT, DeeBERT,
PABEE, KroneckerBERT, multilingual vocabulary work, Bioformer, GPTQ, SmoothQuant, and AWQ.
GPTQ/SmoothQuant/AWQ entries explicitly note that evidence is primarily decoder-only LLMs.

Most links in method metadata and much of the bibliography point to arXiv. Treat an arXiv-only link
as **PREPRINT / NON-PEER-REVIEWED VERSION LINKED** unless you independently cite the final venue.
The repository does not encode peer-review status as a structured field.

## Adding a citation

Add a truthful `Publication` to the method metadata so manifests/CLI expose it, and separately add
the richer bibliography entry if desired. Never use a named paper as method ID unless the code
reproduces that publication's defining procedure. See [extension guide](extending.md).
