"""OpenAI-compatible chat clients (vLLM / DashScope / OpenAI) with round-robin over endpoints."""

from __future__ import annotations

import base64
import itertools
import os
import threading
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional


class EndpointPool:
    """Round-robin over one or more OpenAI-compatible base URLs.

    ``None`` as a base URL means the default OpenAI endpoint. The API key falls
    back to $OPENAI_API_KEY, then to a dummy key (local vLLM servers ignore it).
    """

    def __init__(self, base_urls: Iterable[Optional[str]] = (None,), api_key: Optional[str] = None):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        self._clients = [OpenAI(api_key=key, base_url=url) for url in base_urls]
        self._cycle = itertools.cycle(self._clients)
        self._order = itertools.cycle(range(len(self._clients)))
        self._inflight = [0] * len(self._clients)
        self._lock = threading.Lock()

    def next(self):
        with self._lock:
            return next(self._cycle)

    @contextmanager
    def lease(self, key: Optional[str] = None):
        """A client for one request. With a key, requests for that key go to the same endpoint,
        so prompts sharing a long prefix (one document's section cards, repeated samples) hit
        that server's prefix cache -- unless it has clearly more requests in flight than the
        average, in which case the least busy endpoint takes it (bounded-load affinity)."""
        with self._lock:
            if key is None:
                i = next(self._order)
            else:
                i = zlib.crc32(key.encode("utf-8")) % len(self._clients)
                if self._inflight[i] > 1.25 * sum(self._inflight) / len(self._clients) + 2:
                    i = min(range(len(self._clients)), key=self._inflight.__getitem__)
            self._inflight[i] += 1
        try:
            yield self._clients[i]
        finally:
            with self._lock:
                self._inflight[i] -= 1

    def __len__(self) -> int:
        return len(self._clients)


def image_part(path: str | Path, detail: str = "high") -> Optional[dict]:
    """A base64 image_url content part, or None if the file is missing or unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
    except OSError as e:
        print(f"[WARN] failed to read image {path}: {e}")
        return None
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}", "detail": detail}}
