"""Offline multimodal document tree construction (paper §3.2).

Pipeline, per document:

    MinerU output ──build_tree()──▶ tree_index.json ──summarize──▶ tree_index.json
                   (structure, OCR,                  (page summaries + entities,
                    element text)                     section cards, root summary)
"""

from .mineru import MinerUDocument, build_tree
from .schema import validate_tree

__all__ = ["MinerUDocument", "build_tree", "validate_tree"]
