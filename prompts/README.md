# Prompts

Every prompt template used by AdaNav, grouped by the pipeline stage it belongs to.
Placeholders are written as `{UPPER_CASE}` and filled in by the corresponding agent.

## Offline: tree construction

| File | Produces | Paper reference |
|---|---|---|
| `page_summary_prompt.txt` | Page summary `p_j` — free-text page description, separate table/figure/chart evidence fields, and a categorised entity list (numbers, names, dates, terms, cross-page references) | §3.2, *Hierarchical Summary Generation* |
| `section_summary_prompt.txt` | Section index card `s_i` — topic tags, evidence list with page-level provenance, visual element inventory, consolidated entity list. Deliberately queryable rather than abstractive | §3.2, *Hierarchical Summary Generation* |

Both are prompted with the rendered page image plus the MinerU OCR text
(`{PAGE_TEXT}` / `{PAGE_SUMMARIES}`).

## Online stage 1: routing

| File | Used by | Paper reference |
|---|---|---|
| `agentic_rag/router_prompt.txt` | `f_llm(q, {(t_i, s_i)})` — reads all section titles and summaries jointly, returns the query category and the top-K candidate sections | §3.3, *Router* |
| `agentic_rag/section_filter_prompt.txt` | Section selection as invoked from the agentic router; a variant of the base section filter tuned for the three-way `SPECIFIC / GENERAL / NORMAL` decision | §3.3, *Router* |

The `SPECIFIC` path is decided by `f_rule(q, T)` — pure regular-expression
reference detection — and therefore has no prompt and no LLM cost.

## Online stage 2: navigation

| File | Used by | Paper reference |
|---|---|---|
| `section_filter_prompt.txt` | Coarse section filtering in the single-path (Base) configuration | §3.3, *TreeNav Agent* |
| `page_navigator_prompt.txt` | Page selection inside a candidate section. Also serves as the text-only pre-filter applied to sections longer than `τ_pre` before the VLM does fine-grained visual retrieval | §3.3, *TreeNav Agent* |
| `agentic_rag/general_refilter_prompt.txt` | Precision re-filter over the union of pages returned by the chunked full-document scan; classifies the query as COUNTING vs LOCATING and prunes accordingly | §3.3, *General Agent* |

## Online stage 3: reasoning

| File | Notes |
|---|---|
| `global_reasoner_prompt.txt` | Default reasoner. Scratchpad structure — identify key evidence from both modalities across the retrieved pages, then synthesise the answer. Used for the **AdaNav (Agentic)** rows |
| `global_reasoner_prompt_v2.txt` | Adds financial-ratio and percentage-change conventions plus a strict condition-verification rule |
| `global_reasoner_prompt_v3.txt` | `v2` with the condition-verification rule relaxed, to reduce over-abstention |
| `global_reasoner_prompt_simpledoc.txt` | SimpleDoc-style reasoner prompt, consumes `{DOCUMENT_SUMMARY}`. Used for the **AdaNav (Base)** rows so the comparison with SimpleDoc holds the generation prompt fixed |
| `global_reasoner_reflect_prompt.txt` | Reflection variant: on top of an answer it emits notes and a query update, enabling the iterative-refinement loop |
| `global_reasoner_reflect_prompt_v2.txt` / `_v3.txt` | Reflection counterparts of `v2` / `v3` |

Selection logic in the reference implementation: an explicit `prompt_name`
wins; otherwise `global_reasoner_reflect_prompt` is used when reflection is
enabled and `global_reasoner_prompt` when it is not.

## Reported configurations

| Row in the paper | Reasoner prompt | Iterations |
|---|---|---|
| AdaNav (Base) | `global_reasoner_prompt_simpledoc` | 1 |
| AdaNav (Agentic), Qwen3-VL-30B-A3B | `global_reasoner_prompt` | 3 |
| AdaNav (Agentic), Qwen2.5-VL-32B | `global_reasoner_prompt` | 1 |
