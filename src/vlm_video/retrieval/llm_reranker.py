"""LLM-based reranking using the Anthropic Claude API."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class LLMReranker:
    """Rerank candidates using Claude and return them in relevance order."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return *candidates* sorted by LLM relevance, with updated scores."""
        if not candidates:
            return []
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for LLM reranking.")

        # Inject positional segment_id for any candidate that lacks one
        injected: set[str] = set()
        for i, cand in enumerate(candidates):
            if cand.get("segment_id") is None:
                cand["segment_id"] = i
                injected.add(str(i))

        prompt = self._build_prompt(query, candidates)
        response_text = self._call_claude(prompt)
        try:
            ordered_ids = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Claude response was not valid JSON: {response_text!r}") from exc

        ordered = self._order_candidates(candidates, ordered_ids)
        for i, cand in enumerate(ordered):
            cand["score"] = max(0.0, 1.0 - 0.1 * i)

        # Clean up injected keys so we don't pollute downstream consumers
        for cand in ordered:
            if str(cand.get("segment_id")) in injected:
                del cand["segment_id"]

        return ordered

    def _build_prompt(self, query: str, candidates: list[dict[str, Any]]) -> str:
        lines = []
        for i, cand in enumerate(candidates):
            seg_id = cand.get("segment_id") if cand.get("segment_id") is not None else i
            transcript = cand.get("transcript") or cand.get("asr_text") or ""
            lines.append(f"- {seg_id}: {transcript}")
        segments_block = "\n".join(lines)
        return (
            f"Given the query: '{query}'\n"
            "Rank these video segments from most to least relevant.\n"
            "Return ONLY a JSON array of segment_ids in order, nothing else.\n\n"
            "Segments:\n"
            f"{segments_block}"
        )

    def _call_claude(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        content = parsed.get("content", [])
        if content and isinstance(content, list):
            text = content[0].get("text")
            if text:
                return text.strip()
        raise RuntimeError(f"Unexpected Claude response: {raw}")

    @staticmethod
    def _order_candidates(
        candidates: list[dict[str, Any]],
        ordered_ids: list[Any],
    ) -> list[dict[str, Any]]:
        candidate_map = {str(c.get("segment_id")): dict(c) for c in candidates}
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()

        for seg_id in ordered_ids:
            key = str(seg_id)
            if key in candidate_map and key not in seen:
                ordered.append(candidate_map[key])
                seen.add(key)

        for cand in candidates:
            key = str(cand.get("segment_id"))
            if key not in seen:
                ordered.append(dict(cand))
                seen.add(key)

        return ordered
