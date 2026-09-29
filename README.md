# AdaNav

AdaNav builds a multimodal document tree offline — Document → Section → Page →
Element — and navigates it online at a granularity chosen per question, with no
embedding retrieval model involved. Details are in the
[paper](https://doi.org/10.1007/978-3-032-36023-6_4).

## Release status (Sep 29, 2026)

The document-tree construction and summary generation pipeline is released —
see the [Quickstart](#quickstart-build-the-document-trees) below. Pre-built
trees for MMLongBench-Doc follow once their summaries are regenerated and
verified. The complete release is planned **by October 15, 2026**; until then,
this repository is updated incrementally as each component is verified.

## Release plan

| Target | Deliverable |
|---|---|
| ~~Sep 2026~~ | ~~Prompt templates (all agents + summary generation)~~ — released |
| ~~Sep 29, 2026~~ | ~~Tree construction & summary generation pipeline~~ — released |
| Oct 5, 2026 | Doctree data for MMLongBench-Doc + inference & agent code (Router, TreeNav / General / Specific agents, reasoner) |
| Oct 15, 2026 | Evaluation scripts, remaining benchmark data (PaperTab, FetaTab), reproduction guide |

## Quickstart: build the document trees

Trees are built from [MinerU 2.5](https://github.com/opendatalab/MinerU)
output in two steps: a rule-based structural pass, then summaries from a VLM.
[`data/README.md`](data/README.md) documents each step, the output schema and
the options in full.

**1. Install and configure**

```bash
pip install -r requirements.txt
```

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
