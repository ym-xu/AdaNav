# AdaNav

**Query-Adaptive Multi-granularity Navigation for Long Document Understanding**

Yiming Xu, Eric López, Artemis Llabrés, Maximiliano Hormazábal, Ernest Valveny, Dimosthenis Karatzas
· Computer Vision Center, Universitat Autònoma de Barcelona

ICDAR 2026 · LNCS 16972, pp. 1–18 · [10.1007/978-3-032-36023-6_4](https://doi.org/10.1007/978-3-032-36023-6_4)

---

Long document understanding is hard for VLMs because evidence is scattered
across dozens of pages and heterogeneous content types. Embedding-based
retrieval misses pages whenever the query–evidence relation is compositional;
embedding-free approaches avoid that bias by feeding every page to the VLM, at
high inference cost. Neither exploits the hierarchy that documents already
have.

AdaNav builds a multimodal document tree offline — Document → Section → Page →
Element — and navigates it online at a granularity chosen per question. No
embedding retrieval model is involved. On MMLongBench-Doc it beats open-source
VLM-based agent systems by over 5 accuracy points while sending fewer pages to
the VLM.

## How it works

**Offline — tree construction.** MinerU parses the PDF. When a table of
contents exists, level-1 entries become sections; otherwise section boundaries
are derived from MinerU headings grouped by semantically related titles.
Sections longer than `τ = 30` pages are split by subheading to keep the
downstream navigation budget bounded. Each page node holds the rendered image,
concatenated OCR text with per-block bounding boxes, and a page summary; each
element node holds a layout type, a bbox, block text, a caption, and — for
visual categories above a size threshold — a cropped image. Page and section
summaries are generated with a VLM/LLM using deliberately *queryable*
templates: entity lists, evidence highlights with page provenance, and
verbatim figure/table identifiers rather than abstractive prose.

**Online — three stages.**

1. **Router.** A rule-based detector first pattern-matches explicit page,
   figure, and table references. If any are found the query is `SPECIFIC` and
   the router terminates without an LLM call. Otherwise an LLM reads all
   section titles and summaries jointly and returns either `GENERAL` (needs
   document-wide coverage) or `NORMAL` plus top-K candidate sections.
2. **Specialized navigation.** `TreeNavAgent` (NORMAL) interleaves page images
   and structured summaries per candidate section and asks a VLM for the top
   `B_nav` pages, with a text-only pre-filter for sections above a length
   threshold. `GeneralAgent` (GENERAL) skips section filtering and scans all
   pages in overlapping chunks sized to the token budget. `SpecificAgent`
   (SPECIFIC) resolves references through a cascaded lookup over three
   offline-built indices — logical-to-physical page map, caption/block element
   index, page-level entity index — at zero LLM cost.
3. **Reasoner.** A VLM answers from the raw retrieved pages. The retrieval
   indices are used for navigation only and are deliberately *not* forwarded,
   so the reasoner grounds its answer in the original visual layout.

## Results

End-to-end accuracy (%), judged by GPT-4.1. *Pg. Ret.* is the mean number of
pages fed to the VLM per question.

| Method | Pg. Ret. | MMLB | PaperTab | FetaTab | Avg. |
|---|---|---|---|---|---|
| Qwen2.5-VL-32B (no retrieval) | – | 22.18 | 7.12 | 16.14 | 15.15 |
| Qwen2.5-VL-32B + GT pages (oracle) | – | 67.94 | – | – | – |
| Qwen3-VL-30B-A3B + GT pages (oracle) | – | 67.65 | – | – | – |
| M3DocRAG | 6 | 41.80 | 60.10 | 79.80 | 60.57 |
| MDocAgent | 12 | 55.30 | **64.90** | **84.50** | 68.23 |
| SimpleDoc | 3.2 | 59.55 | 64.38 | 80.31 | 68.08 |
| **AdaNav** (Qwen2.5-VL-32B) | 4.17 | 59.46 | 61.06 | 82.89 | 67.80 |
| **AdaNav** (Qwen3-VL-30B-A3B) | 4.17 | **65.73** | 63.61 | 83.75 | **71.02** |

Baselines all use ColQwen-2.5 for visual embedding retrieval and
Qwen2.5-VL-32B-Instruct for generation; their numbers are taken from SimpleDoc.

Retrieval on MMLongBench-Doc (n = 847 questions with annotated evidence pages):

| Method | Avg. pages | All-Hit (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|
| ColQwen-2.5 (k = 2) | 2 | 64.12 | – | 38.75 |
| ColQwen-2.5 (k = 10) | 10 | 83.60 | – | 18.38 |
| SimpleDoc (top-30) | 3.46 | 67.37 | – | 62.22 |
| AdaNav (Base) | 5.94 | 77.5 | 79.6 | 48.9 |
| **AdaNav (Agentic)** | **4.17** | **80.2** | **85.5** | 57.1 |

Increasing `k` for embedding retrieval trades All-Hit against precision —
38.75 → 18.38 F1 from k = 2 to k = 10. Tree navigation reaches ColQwen's k = 10
coverage on 4.17 pages instead of 10.

Ablation on MMLongBench-Doc (Qwen3-VL-30B-A3B), showing which agent handles
each query category:

| Variant | Spec. | Gen. | Norm. | Recall | F1 | ACC |
|---|---|---|---|---|---|---|
| AdaNav (Base) | TreeNav | TreeNav | TreeNav | 79.6 | 48.9 | 61.4 |
| w/o SpecificAgent | TreeNav | General | TreeNav | 83.9 | 56.2 | 63.4 |
| w/o GeneralAgent | Specific | TreeNav | TreeNav | 83.2 | 53.7 | 62.8 |
| **AdaNav (Agentic)** | Specific | General | TreeNav | **85.2** | **57.1** | **65.7** |

Router decisions over the 1,073 MMLongBench-Doc questions: `NORMAL` 823
(76.7%), `GENERAL` 137 (12.8%), `SPECIFIC` 113 (10.5%). The `SPECIFIC` path
removes LLM overhead for more than one question in ten.

## Repository status

This repository is being assembled incrementally from the research working
tree. What has landed so far:

```
AdaNav/
├── prompts/                    # every prompt template, with a paper mapping
│   ├── README.md
│   ├── page_summary_prompt.txt
│   ├── section_summary_prompt.txt
│   ├── section_filter_prompt.txt
│   ├── page_navigator_prompt.txt
│   ├── global_reasoner_prompt*.txt
│   ├── global_reasoner_reflect_prompt*.txt
│   └── agentic_rag/            # router, section filter, general re-filter
│       └── ...
├── data/
│   ├── README.md               # schema, statistics, how to get page images
│   └── doctree/mmlongbench/    # 134 pre-built document trees
│       └── <doc_id>/tree_index.json
└── scripts/
    └── export_doctree.py       # working tree -> portable release layout
```

| Component | State |
|---|---|
| Prompts | ✅ complete |
| Doctree data (MMLongBench-Doc) | ⚠️ incomplete — structure only, summaries not yet included ([details](data/README.md)) |
| Doctree data (PaperTab, FetaTab) | ⏳ not yet exported |
| Ingestion code (MinerU adapter, tree builder, summary generation) | ⏳ pending migration |
| Inference code (router, three navigation agents, reasoner) | ⏳ pending migration |
| Evaluation scripts | ⏳ pending migration |

> **Note.** The doctree data currently released is a structural snapshot: the
> hierarchy, OCR text and element bounding boxes are present, but the page and
> section summaries that navigation depends on are not yet included, so the
> pipeline will not run on it as-is. See [`data/README.md`](data/README.md).

## Reproduction setup

Backbone VLM: Qwen3-VL-30B-A3B for the primary results, Qwen2.5-VL-32B-Instruct
for the matched-backbone comparison against baselines. The tree index is built
offline once per document. At inference, per-section page budget `B_nav = 3` and
general-agent budget `B_gen = 2` pages per section. Answers are judged by GPT-4.1
at temperature 0, following SimpleDoc's protocol.

Benchmarks: [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc)
— 1,073 questions over 134 documents, from the 1,082 / 135 the benchmark ships
— plus PaperTab (393 / 307) and FetaTab (1,023 / 878).

## Citation

```bibtex
@inproceedings{xu2026adanav,
  title     = {AdaNav: Query-Adaptive Multi-granularity Navigation for Long Document Understanding},
  author    = {Xu, Yiming and L{\'o}pez, Eric and Llabr{\'e}s, Artemis and
               Hormaz{\'a}bal, Maximiliano and Valveny, Ernest and Karatzas, Dimosthenis},
  booktitle = {Document Analysis and Recognition -- ICDAR 2026},
  series    = {Lecture Notes in Computer Science},
  volume    = {16972},
  pages     = {1--18},
  year      = {2027},
  publisher = {Springer},
  doi       = {10.1007/978-3-032-36023-6_4}
}
```

Contact: yiming@cvc.uab.es
