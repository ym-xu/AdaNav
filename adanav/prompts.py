"""Prompt templates in prompts/ (see prompts/README.md)."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load prompts/<name>.txt; ``name`` may include a subdirectory, e.g. "agentic_rag/router_prompt"."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
