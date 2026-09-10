# AdaNav

**Query-Adaptive Multi-granularity Navigation for Long Document Understanding**

Yiming Xu, Eric López, Artemis Llabrés, Maximiliano Hormazábal, Ernest Valveny,
Dimosthenis Karatzas · Computer Vision Center, Universitat Autònoma de Barcelona

ICDAR 2026 · LNCS 16972, pp. 1–18 · [10.1007/978-3-032-36023-6_4](https://doi.org/10.1007/978-3-032-36023-6_4)

---

AdaNav builds a multimodal document tree offline — Document → Section → Page →
Element — and navigates it online at a granularity chosen per question, with no
embedding retrieval model involved. Details are in the
[paper](https://doi.org/10.1007/978-3-032-36023-6_4).

## Release status (Sep 10, 2026)

The code and data are being cleaned up and refactored from the research
codebase for public release. The complete release is planned **by
October 15, 2026**; until then, this repository is updated incrementally as
each component is verified.

## Release plan

| Target | Deliverable |
|---|---|
| ~~Sep 2026~~ | ~~Prompt templates (all agents + summary generation)~~ — released |
| Sep 25, 2026 | Doctree data for MMLongBench-Doc + inference & agent code (Router, TreeNav / General / Specific agents, reasoner) |
| Oct 5, 2026 | Tree construction & summary generation pipeline |
| Oct 15, 2026 | Evaluation scripts, remaining benchmark data (PaperTab, FetaTab), reproduction guide |

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
