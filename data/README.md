# Doctree data

Multimodal document trees — the offline artefact described in §3.2 of the paper.
One directory per document:

```
data/doctree/<dataset>/
├── page_summaries.json        # page summary cache (step 3)
└── <doc_id>/
    └── tree_index.json
```

## Building

Trees are built from [MinerU 2.5](https://github.com/opendatalab/MinerU) output
in three steps. Steps 1–2 are rule-based and take seconds; step 3 calls a VLM.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1. Prepare (PaperTab / FetaTab only) | `prepare_mineru.py` | source PDF, MinerU content list | `images/page_N.png`, `process/toc.json`, `process/logical_pages.json` in the MinerU directory |
| 2. Structure | `build_doctree.py` | MinerU directory | `data/doctree/<dataset>/<doc_id>/tree_index.json` without summaries |
| 3. Summaries | `summarize_doctree.py` | the trees + page images | the same trees, now with summaries; `page_summaries.json` cache |

### Setup

```bash
pip install -r requirements.txt   # PyMuPDF is only needed for rendering in step 1
```

Edit [`configs/datasets.yaml`](../configs/datasets.yaml): set `mineru_root` to
the directory holding one MinerU output directory per document (or do so in a
git-ignored copy, `configs/datasets.local.yaml`, which is read instead). `doctree_dir`
(default `data/doctree/<dataset>`) is where trees go. The same file holds the
section-splitting settings (τ = 30 pages).

Try everything on a couple of documents first by adding `--first 2` (or
`--doc <doc_id>`) to each command.

### Step 1: prepare PaperTab / FetaTab

MMLongBench-Doc's MinerU output already contains page images,
`process/toc.json` (level-1 headings with logical page labels) and
`process/logical_pages.json` (logical → physical page map); skip this step for
it. The tool that produced those two files is not part of this repository.

PaperTab and FetaTab need both:

```bash
python scripts/prepare_mineru.py --dataset papertab --render --toc
python scripts/prepare_mineru.py --dataset fetatab  --render --toc
```

`--toc` levels MinerU's headings by numbering scheme for papers and by heading
height for Wikipedia pages. Existing files are kept unless `--force`.

### Step 2: structure

```bash
python scripts/build_doctree.py --dataset mmlongbench
```

This creates sections from the TOC (a "Preface" covers pages before the first
heading; adjacent sections share their boundary page; sections over 30 pages are
split on sub-headings or into 25-page chunks), attaches MinerU elements to
pages, and adds table bodies and figure / table captions to the element and page
text. The output reproduces the section structure of the trees evaluated in the
paper field for field.

Existing trees are not overwritten unless `--force` — it would drop their
summaries. A `[WARN]` line reports a structural oddity; for MMLongBench-Doc
expect exactly three, for the documents with out-of-order TOCs listed below.

### Step 3: summaries

Serve a VLM behind an OpenAI-compatible API. The paper uses Qwen3-VL-30B-A3B
(about 60 GB of weights in bf16: one 80 GB GPU, or add `--tensor-parallel-size 2`):

```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 8011
# optional second replica on another GPU
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 8012
```

Then:

```bash
python scripts/summarize_doctree.py --dataset mmlongbench \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --base-url http://127.0.0.1:8011/v1 --base-url http://127.0.0.1:8012/v1
```

Requests are spread round-robin over every `--base-url`. For each document it
summarises every page (image + OCR text → `summary`, `entities`), then writes a
card per section from its page summaries, then joins the cards into
`root_summary`. MMLongBench-Doc has about 6,500 pages; one server takes roughly
four hours.

**Resuming.** Page summaries are cached in
`data/doctree/<dataset>/page_summaries.json` as they arrive (saved every 50
pages and at the end). Re-running the same command skips complete trees, serves
cached pages without a model call, and retries only what is missing: pages whose
request failed are never cached, and section cards are regenerated for any tree
that still has a gap.

**Checking.** A tree is complete when `root_summary` and every section and page
`summary` are non-empty. `[ERROR]` lines name pages or sections whose request
failed; re-run the same command until none remain.

The model samples at temperature 0.1, so regenerated summaries differ in wording
from the ones used in the paper.

| Option | Effect |
|---|---|
| `--first N`, `--doc ID` | limit to some documents (`--doc` repeatable) |
| `--force` | redo complete trees too (cached pages are still reused; add `--no-cache` for fresh page summaries) |
| `--page-only` | page summaries only |
| `--section-only` | rebuild section cards and root summary from existing page summaries |
| `--refresh-refs` | recompute the `VISUAL_REFS_FROM_PAGES` block of each card; no model calls |
| `--no-cache`, `--cache PATH` | disable, or relocate, the page summary cache |
| `--workers`, `--doc-workers` | concurrent page requests per document (4), documents in parallel (2) |
| `--api-key` | key for hosted endpoints (default `$OPENAI_API_KEY`) |

## MMLongBench-Doc, structure after step 2

| | |
|---|---|
| Documents | 134 |
| Sections | 2,902 (at most 30 pages each; adjacent sections share their boundary page) |
| Pages | 6,492 (mean 48.4 per document, range 9–468) |
| Elements | 78,613 |
| Size on disk | 40 MB (compact JSON, before summaries) |

Element type distribution: `text` 51,468 · `image` 7,478 · `header` 4,948 ·
`page_number` 4,452 · `list` 4,142 · `footer` 3,258 · `table` 1,629 ·
`code` 646 · `aside_text` 235 · `page_footnote` 224 · `equation` 119 ·
`ref_text` 14.

MMLongBench-Doc ships 135 documents and 1,082 questions; the 134 documents here
account for exactly the 1,073 questions the paper evaluates on (`mi_phone`,
holding the other 9, is excluded from the reported results too).

Three documents (`efis-140411041451-phpapp01_95`,
`mmdetection-readthedocs-io-en-v2.18.0`, `san-francisco-11-contents`) have a
TOC whose headings are out of page order; their sections are kept as the
evaluated trees had them, and `build_doctree.py` prints a warning.

## Schema

```jsonc
{
  "doc_id": "2307.09288v2",
  "source_dir": "",
  "num_pages": 77,
  "root_summary": "",            // section cards joined by "\n\n---\n\n"
  "sections": [
    {
      "id": "sec_00",
      "title": "Preface",        // pages before the first heading
      "start_page": 0,           // 0-indexed, inclusive
      "end_page": 1,             // 0-indexed, inclusive; = next section's start_page
      "summary": "",             // section index card s_i
      "pages": [
        {
          "page_idx": 0,
          "image_path": "images/page_1.png",   // relative to <mineru_root>/<doc_id>/
          "ocr_text": "LLAMA 2: Open Foundation...",
          "summary": "",                        // page summary p_j
          "entities": "",                       // categorised entities (added by step 3)
          "elements": [
            {
              "id": "elem_0_0",
              "type": "text",                   // MinerU layout category
              "page_idx": 0,
              "bbox": [112, 237, 274, 250],     // [x0, y0, x1, y1] on the source page
              "text": "FOR RELEASE MAY 3, 2018",
              "path": "",                       // MinerU crop, relative like image_path
              "caption": ""                     // "" or MinerU's list of caption lines
            }
          ]
        }
      ]
    }
  ]
}
```

A boundary page appears in both sections that share it, as an identical copy.

For `table` and `image` elements, `text` holds the linearised table body
(`| cell | cell` rows) with `Table:` / `Figure:` caption and `Note:` footnote
lines, and the same text is appended to the page's `ocr_text`.

`type` takes the MinerU layout categories listed above. The paper's element
node `E_k = (type_k, bbox_k, I_k^crop)` maps onto `type` / `bbox` / `path`.

## Page images

Page images and source PDFs are **not** redistributed — MMLongBench-Doc
documents carry their own licences. `image_path` resolves against
`<mineru_root>/<doc_id>/`. Get the documents from
[MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc) and run MinerU
2.5 over them.

## Other benchmarks

The paper also evaluates on PaperTab (393 questions, 307 documents) and FetaTab
(1,023 questions, 878 documents). Their TOCs are derived from MinerU headings
in step 1.
