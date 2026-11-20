# AdaNav

AdaNav builds a multimodal document tree offline — Document → Section → Page →
Element — and navigates it online at a granularity chosen per question, with no
embedding retrieval model involved. Details are in the
[paper](https://doi.org/10.1007/978-3-032-36023-6_4).

This repository contains the document-tree construction and summary generation
pipeline, the online agents (router, Specific / General / TreeNav navigators,
reasoner), the evaluation scripts, and the prompt templates they use.

## Repository layout

```
adanav/doctree/   tree construction from MinerU output, TOC derivation, summaries
adanav/agents/    online inference: router, navigators, reasoner, pipeline
adanav/eval/      benchmark loading, answer judge, retrieval and QA metrics
adanav/prompts.py prompt loading
adanav/llm.py     OpenAI-compatible client
adanav/runs.py    run records (git state, command line) shared by the scripts
configs/          dataset paths and section-splitting settings
prompts/          prompt templates (see prompts/README.md)
scripts/          command-line entry points
data/             document trees; MMLongBench-Doc's are included (see data/README.md)
tests/            unit tests
```

## Installation

```bash
git clone https://github.com/ym-xu/AdaNav.git
cd AdaNav
pip install -r requirements.txt
```

PyMuPDF is only needed to render page images (PaperTab and FetaTab, `scripts/render_pages.py`).

## Building document trees

The trees for MMLongBench-Doc are included in
[`data/doctree/mmlongbench/`](data/doctree/mmlongbench/), with summaries, printed page
labels and figure descriptions; skip to [Answering questions](#answering-questions) to use
them. Other datasets are built from [MinerU 2.5](https://github.com/opendatalab/MinerU)
output: a rule-based structural pass, then summaries from a VLM.
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

## Answering questions

Each question is routed (explicit page / figure / table reference, global question, or
content lookup), navigated to a few pages, and answered from the page images and OCR
text. Render the page images from the
[MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc) PDFs: 150 DPI for
navigation (the same images as MinerU's; skip this if you have MinerU output) and 200 DPI
for the reasoner. Set `mineru_root` to the first directory and `samples` to the
benchmark's `samples.json` in the datasets config, serve the model with a 128k context
(`vllm serve ... --max-model-len 131072`), then run:

```bash
python scripts/render_pages.py --dataset mmlongbench --dpi 150 --png \
    --pdf-dir /path/to/MMLongBench-Doc/documents --out data/images/mmlongbench
python scripts/render_pages.py --dataset mmlongbench --dpi 200 \
    --pdf-dir /path/to/MMLongBench-Doc/documents --out data/images/mmlongbench_200dpi

python scripts/run_adanav.py --dataset mmlongbench --run-dir results/mmlongbench/agentic \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct --base-url http://127.0.0.1:8011/v1 \
    --page-images data/images/mmlongbench_200dpi

# retrieval metrics, routing, and accuracy judged by GPT-4.1 ($OPENAI_API_KEY)
python scripts/evaluate_adanav.py --run-dir results/mmlongbench/agentic
```

`--preset base` runs AdaNav (Base); `--iterations 1` stops after the first reasoner
round (the agentic preset allows 3); `--no-specific`, `--no-general` and `--no-refilter`
switch off one component; without `--page-images` the reasoner reads the MinerU page images;
`--retrieval-only` stops before the reasoner. `scripts/judge_batch.py` judges the same
answers through the OpenAI Batch API at half the price.

## Prompts

[`prompts/`](prompts/) holds the templates for tree construction (page and
section summaries), routing, navigation and reasoning.
[`prompts/README.md`](prompts/README.md) maps each file to its stage and the
section of the paper it implements.

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
