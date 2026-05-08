"""Result ranking / re-ranking utilities."""

from __future__ import annotations

import re
from typing import Any

KEYWORD_DIVISOR: float = 3.0


def _keyword_score(query_text: str, transcript: str) -> float:
    if not transcript:
        return 0.0
    words = {w for w in re.findall(r"\w+", query_text.lower()) if w}
    if not words:
        return 0.0
    transcript_lower = transcript.lower()
    hits = sum(1 for w in words if w in transcript_lower)
    return min(hits / KEYWORD_DIVISOR, 1.0)


def rerank_results(
    results: list[dict[str, Any]],
    query_text: str | None = None,
) -> list[dict[str, Any]]:
    """Re-rank retrieval results using a hybrid semantic + keyword score.

    Parameters
    ----------
    results:
        Ordered list of result dicts as returned by
        :meth:`~vlm_video.retrieval.base_index.BaseIndex.search`.
    query_text:
        Original query string.  Reserved for future lexical / semantic
        re-ranking implementations.

    Returns
    -------
    list[dict]
        Re-ranked result list (currently identical to *results*).
    """
    if not query_text:
        return list(results)

    scored: list[dict[str, Any]] = []
    for res in results:
        entry = dict(res)
        semantic_score = float(entry.get("score", 0.0))
        transcript = entry.get("transcript") or entry.get("asr_text") or ""
        keyword_score = _keyword_score(query_text, transcript)
        entry["semantic_score"] = semantic_score
        entry["keyword_score"] = keyword_score
        entry["score"] = 0.6 * semantic_score + 0.4 * keyword_score
        scored.append(entry)

    return sorted(scored, key=lambda r: r.get("score", 0.0), reverse=True)
