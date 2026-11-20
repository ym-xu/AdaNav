"""Binary answer correctness judged by an OpenAI model, with SimpleDoc's evaluation prompt.

The prompt is SimpleDoc's (evaluation/evaluate_responses.py) word for word; the paper
uses GPT-4.1 at temperature 0. Any OpenAI-compatible endpoint can judge (e.g. a local
vLLM model), so scores are only comparable under the same judge. Answers equal after
light normalisation ("41%" vs "41", "1,234" vs "1234") are scored 1 without a model call.
"""

from __future__ import annotations

import json
import re
from typing import Tuple

PROMPT = """Question: {question}
Predicted Answer: {pred}
Ground Truth Answer: {gt}

Please evaluate if the predicted answer is correct compared to the ground truth.
Score the answer on:
Binary correctness (0-1): 1 if the answer is correct, 0 if it is incorrect

Return only a string with these scores in a dictionary and can be parsed by json.loads, e.g. {{"binary_correctness": 1}}"""


def normalize(text: str) -> str:
    t = text.strip().strip(".,;:!? ")
    t = re.sub(r"^([\d.]+)\s*%$", r"\1", t)
    return t.replace(",", "").lower()


def parse_score(text: str) -> int:
    try:
        return int(json.loads(text).get("binary_correctness", 0))
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        m = re.search(r'"?binary_correctness"?\s*:\s*([01])', text)
        if m:
            return int(m.group(1))
        raise ValueError(f"unparseable judge response: {text[:200]!r}")


class Judge:
    def __init__(self, model: str = "gpt-4.1", pool=None):
        """``pool``: an EndpointPool (default: the OpenAI API, key from $OPENAI_API_KEY)."""
        if pool is None:
            from adanav.llm import EndpointPool
            pool = EndpointPool([None])
        self.model, self.pool = model, pool

    def score(self, question: str, pred: str, gt: str) -> Tuple[int, str]:
        """(score, how): how is "exact" or the judge's raw response."""
        if normalize(pred) == normalize(gt):
            return 1, "exact"
        response = self.pool.next().chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=512,
            messages=[{"role": "user", "content": PROMPT.format(question=question, pred=pred, gt=gt)}])
        text = (response.choices[0].message.content or "").strip()
        return parse_score(text), text
