"""FAISS retrieval index (requires ``pip install vlm_video[faiss]``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore[import-untyped]
except ImportError as _faiss_err:
    raise ImportError(
        "faiss is required for FaissIndex but is not installed.\n"
        "Install FAISS: pip install vlm_video[faiss]\n"
        "Note: faiss-cpu is supported on Linux and macOS. "
        "On Windows, use WSL or a conda environment: conda install -c conda-forge faiss-cpu"
    ) from _faiss_err

from vlm_video.retrieval.base_index import BaseIndex


class FaissIndex(BaseIndex):
    """FAISS-backed retrieval index for fast approximate nearest-neighbour search.

    Parameters
    ----------
    index_type:
        FAISS index factory string, e.g. ``"IndexFlatIP"`` (exact inner product)
        or ``"IVF256,Flat"`` for an IVF index.
    nprobe:
        Number of IVF cells to visit during search (only relevant for IVF
        indexes; ignored for flat indexes).
    """

    def __init__(
        self,
        index_type: str = "IndexFlatIP",
        nprobe: int = 10,
    ) -> None:
        self.index_type = index_type
        self.nprobe = nprobe
        self._index: faiss.Index | None = None
        self._metadata: list[dict[str, Any]] = []
        self._dim: int = 0

    # ── BaseIndex interface ───────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Build a FAISS index from *embeddings*.

        Embeddings are L2-normalised before insertion so that inner-product
        search equals cosine similarity (when using ``IndexFlatIP``).
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings and metadata length mismatch: "
                f"{len(embeddings)} vs {len(metadata)}"
            )

        vectors = embeddings.astype(np.float32)
        faiss.normalize_L2(vectors)
        self._dim = vectors.shape[1]

        self._index = faiss.index_factory(self._dim, self.index_type)
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self.nprobe

        if not self._index.is_trained:
            self._index.train(vectors)

        self._index.add(vectors)
        self._metadata = list(metadata)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k results sorted by descending cosine similarity."""
        if self._index is None or self._index.ntotal == 0:
            return []

        q = query_emb.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(q)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self._metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results

    def save(self, path: str | Path) -> None:
        """Save FAISS index and metadata to *path* directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(path / "faiss.index"))
        with (path / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> None:
        """Load FAISS index and metadata from *path* directory."""
        path = Path(path)
        index_file = path / "faiss.index"
        meta_file = path / "metadata.json"

        if not index_file.exists():
            raise FileNotFoundError(f"faiss.index not found in {path}")
        if not meta_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {path}")

        self._index = faiss.read_index(str(index_file))
        with meta_file.open("r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

    def __len__(self) -> int:
        return self._index.ntotal if self._index is not None else 0
