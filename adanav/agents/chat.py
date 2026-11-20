"""Model calls shared by all agents, and the per-question trace they write to."""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Reply:
    text: str
    finish_reason: Optional[str]
    tokens: Dict[str, int]


@dataclass
class Trace:
    """Everything one question did: every model call (with its raw response) and every fallback.

    Agents are stateless; they append here instead. Thread-safe, since sections are
    navigated in parallel.
    """

    events: List[Dict[str, Any]] = field(default_factory=list)
    key: Optional[str] = None  # endpoint affinity (the document id): see EndpointPool.lease
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **event: Any) -> None:
        with self._lock:
            self.events.append(event)

    def fallback(self, step: str, reason: str, **extra: Any) -> None:
        self.add(kind="fallback", step=step, reason=reason, **extra)

    def calls(self, step: Optional[str] = None) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == "call" and (step is None or e["step"] == step)]

    def summary(self) -> Dict[str, int]:
        """Counts per step: calls, truncated responses, fallbacks, prompt/completion tokens."""
        c: Counter = Counter()
        for e in self.events:
            if e["kind"] == "call":
                c[f"{e['step']}_calls"] += 1
                c[f"{e['step']}_truncated"] += e.get("finish_reason") == "length"
                c["prompt_tokens"] += e.get("tokens", {}).get("prompt", 0)
                c["completion_tokens"] += e.get("tokens", {}).get("completion", 0)
            else:
                c[f"{e['step']}_fallbacks"] += 1
        return {k: v for k, v in c.items() if v}


class ChatModel:
    """One model behind an EndpointPool. Each call is retried on request errors and recorded in the trace.

    The prompt is not recorded (it is rebuilt from the tree and the prompt file); the
    response, finish reason and token usage are.
    """

    def __init__(self, model: str, pool, temperature: float = 0.1, retries: int = 3):
        self.model, self.pool, self.temperature, self.retries = model, pool, temperature, retries

    def __call__(self, content, *, step: str, max_tokens: int, trace: Trace, temperature: Optional[float] = None,
                 **meta: Any) -> Reply:
        messages = [{"role": "user", "content": content}]
        temperature = self.temperature if temperature is None else temperature
        for attempt in range(self.retries + 1):
            try:
                if hasattr(self.pool, "lease"):
                    with self.pool.lease(trace.key) as client:
                        response = client.chat.completions.create(
                            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature)
                else:
                    response = self.pool.next().chat.completions.create(
                        model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature)
                break
            except Exception as e:  # network / server errors: back off, then give up
                if attempt == self.retries:
                    trace.add(kind="call", step=step, error=repr(e), **meta)
                    raise
                time.sleep(2 ** attempt)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        reply = Reply(text=choice.message.content or "",
                      finish_reason=getattr(choice, "finish_reason", None),
                      tokens={"prompt": usage.prompt_tokens, "completion": usage.completion_tokens} if usage else {})
        trace.add(kind="call", step=step, finish_reason=reply.finish_reason, tokens=reply.tokens,
                  response=reply.text, **meta)
        return reply


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for budgeting prompt sizes (about 3 characters per token)."""
    return len(text) // 3 + 1
