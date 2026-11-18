# AdaNav

AdaNav builds a multimodal document tree offline — Document → Section → Page →
Element — and navigates it online at a granularity chosen per question, with no
embedding retrieval model involved. Details are in the
[paper](https://doi.org/10.1007/978-3-032-36023-6_4).

This repository contains the document-tree construction and summary generation
pipeline, and the prompt templates used by every AdaNav agent.

## Repository layout

```
adanav/doctree/   tree construction from MinerU output, TOC derivation, summaries
adanav/prompts.py prompt loading
adanav/llm.py     OpenAI-compatible client
configs/          dataset paths and section-splitting settings
prompts/          prompt templates (see prompts/README.md)
scripts/          command-line entry points
data/             output location for the trees (see data/README.md)
tests/            unit tests
```

## Installation

```bash
git clone https://github.com/ym-xu/AdaNav.git
cd AdaNav
pip install -r requirements.txt
```

PyMuPDF is only needed to render page images for PaperTab and FetaTab.

## Building document trees

Trees are built from [MinerU 2.5](https://github.com/opendatalab/MinerU)
output in two steps: a rule-based structural pass, then summaries from a VLM.
[`data/README.md`](data/README.md) documents each step, the output schema and
the options in full.

**1. Configure**

Set `mineru_root` for each dataset in
[`configs/datasets.yaml`](configs/datasets.yaml) to your MinerU output
(one directory per document), or put your paths in a git-ignored copy,
`configs/datasets.local.yaml`, which is read instead. Trees are written to
`data/doctree/<dataset>/`.

**2. Build the structure** (seconds, no model)

```bash
# PaperTab / FetaTab only: render page images and derive a TOC first
python scripts/prepare_mineru.py --dataset papertab --render --toc

python scripts/build_doctree.py --dataset mmlongbench
```

**3. Generate summaries** (needs an OpenAI-compatible VLM endpoint)

```bash
# e.g. serve the paper's backbone with vLLM
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 8011

python scripts/summarize_doctree.py --dataset mmlongbench \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct --base-url http://127.0.0.1:8011/v1
```

MMLongBench-Doc has about 6,500 pages; one server takes roughly four hours.
The run is resumable — see [Step 3: summaries](data/README.md#step-3-summaries).

Add `--first 2` or `--doc <doc_id>` to any command to try it on a few
documents first.

## Prompts

[`prompts/`](prompts/) holds the templates for tree construction (page and
section summaries), routing, navigation and reasoning.
[`prompts/README.md`](prompts/README.md) maps each file to its stage, the
section of the paper it implements, and the configurations reported in the
paper.

## Tests

```bash
python -m unittest discover tests
```

## Citation

```bibtex
@inproceedings{xu2026adanav,
  title     = {AdaNav: Query-Adaptive Multi-granularity Navigation
               for Long Document Understanding},
  author    = {Xu, Yiming and L{\'o}pez, Eric and Llabr{\'e}s, Artemis and
               Hormaz{\'a}bal, Maximiliano and Valveny, Ernest and
               Karatzas, Dimosthenis},
  booktitle = {International Conference on Document Analysis and Recognition
               (ICDAR)},
  series    = {LNCS},
  volume    = {16972},
  pages     = {1--18},
  year      = {2026},
  doi       = {10.1007/978-3-032-36023-6_4},
}
```
