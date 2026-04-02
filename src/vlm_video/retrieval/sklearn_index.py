"""Scikit-learn cosine-similarity retrieval index (no optional dependencies)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from vlm_video.retrieval.base_index import BaseIndex


class SklearnIndex(BaseIndex):
    """In-memory retrieval index backed by scikit-learn cosine similarity.

    This implementation requires no optional dependencies and works on
    both CPU and any platform.

    Parameters
    ----------
    metric:
        Currently only ``"cosine"`` is supported (default).
    """

    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] = []

    # ── BaseIndex interface ───────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Populate the index.

        Parameters
        ----------
        embeddings:
            Float32 array of shape ``(N, D)``.
        metadata:
            Parallel list of N metadata dicts.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings and metadata length mismatch: "
                f"{len(embeddings)} vs {len(metadata)}"
            )
        self._embeddings = embeddings.astype(np.float32)
        self._metadata = list(metadata)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k results sorted by descending cosine similarity.

        Parameters
        ----------
        query_emb:
            1-D query vector of shape ``(D,)``.
        top_k:
            Number of results to return.

        Returns
        -------
        list[dict]
            Each dict contains all metadata keys plus ``"score"`` (float).
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        q = query_emb.astype(np.float32).reshape(1, -1)
        sims = cosine_similarity(q, self._embeddings)[0]  # shape (N,)

        k = min(top_k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            entry = dict(self._metadata[idx])
            entry["score"] = float(sims[idx])
            results.append(entry)

        return results

    def save(self, path: str | Path) -> None:
        """Save index to *path* directory.

        Writes two files:

        * ``embeddings.npy``
        * ``metadata.json``
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._embeddings is not None:
            np.save(path / "embeddings.npy", self._embeddings)
        with (path / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> None:
        """Load index from *path* directory."""
        path = Path(path)
        emb_file = path / "embeddings.npy"
        meta_file = path / "metadata.json"

        if not emb_file.exists():
            raise FileNotFoundError(f"embeddings.npy not found in {path}")
        if not meta_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {path}")

        self._embeddings = np.load(emb_file)
        with meta_file.open("r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

    def __len__(self) -> int:
        return len(self._embeddings) if self._embeddings is not None else 0
