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
| `agentic_rag/section_filter_prompt.txt` | `f_llm(q, {(t_i, s_i)})` in the agentic preset: reads the section titles and index cards, returns the top-K sections, or `ALL` for a global question (the GENERAL flag) | §3.3, *Router* |
| `section_filter_prompt.txt` | The same step in the base preset (no rule stage, so it also covers explicit page references) | §3.3 |
| `section_filter_chunk_prompt.txt` | Recall pass when a document's cards do not fit one prompt: each chunk of sections returns up to 3 candidates, or `ALL` | §3.3 |
| `section_rerank_prompt.txt` | Picks the top K among the candidates of all chunks | §3.3 |

The `SPECIFIC` path is decided by `f_rule(q, T)` — regular-expression reference
detection and index lookup in `adanav/agents/specific.py` — and therefore has no prompt
and no LLM cost.

## Online stage 2: navigation

| File | Used by | Paper reference |
|---|---|---|
| `vision_page_navigator_prompt.txt` | `π_page` (the default): page summaries and entities interleaved with the page images (`{PAGES}` marks where they go), sampled 3 times and merged; shared by the TreeNav Agent (`B_nav` = 4 pages per section) and the General Agent (`B_gen` = 10) | §3.4 |
| `vision_prefilter_prompt.txt` | Text-only narrowing of sections longer than 30 pages before the vision navigator sees them | §3.4 |
| `page_navigator_prompt.txt` | Text-only `π_page` (`--page-navigator text`): page summaries, entities and OCR text only | §3.4 |
| `agentic_rag/general_refilter_prompt.txt` | Refilter over the union of the General Agent's pages; classifies the question as COUNTING vs LOCATING and keeps up to 15 / 5 pages | §3.4, §4.4 |

## Online stage 3: reasoning

| File | Notes |
|---|---|
| `global_reasoner_prompt.txt` | Reasoner of the agentic preset (one round). Scratchpad, then `<answer>` or `<not_answerable>` |
| `global_reasoner_prompt_simpledoc.txt` | SimpleDoc-style reasoner, consumes `{DOCUMENT_SUMMARY}`; reasoner of the base preset |
| `global_reasoner_reflect_prompt.txt` | Used when more than one round is allowed (`--iterations > 1`): may answer `<query_update>` + `<notes>`, and the updated query is navigated again |

`scripts/run_adanav.py --reasoner-prompt <name>` overrides the reasoner prompt of a preset.
