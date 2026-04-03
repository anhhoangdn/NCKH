"""Abstract base class for retrieval indexes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseIndex(ABC):
    """Interface that all retrieval index implementations must satisfy."""

    @abstractmethod
    def build(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Populate the index from pre-computed embeddings.

        Parameters
        ----------
        embeddings:
            Array of shape ``(N, D)`` containing segment embeddings.
        metadata:
            List of N metadata dicts (one per embedding), e.g. segment info.
        """

    @abstractmethod
    def search(self, query_emb: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the *top_k* most similar entries.

        Parameters
        ----------
        query_emb:
            1-D query embedding vector of shape ``(D,)``.
        top_k:
            Number of results to return.

        Returns
        -------
        list[dict]
            Each result dict contains at least ``score`` (float) and all
            keys from the corresponding metadata entry.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the index to *path* (directory or file prefix).

        Parameters
        ----------
        path:
            Destination path.  The implementation decides the exact file
            layout (e.g. ``path/embeddings.npy`` + ``path/metadata.json``).
        """

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load a previously saved index from *path*.

        Parameters
        ----------
        path:
            Source path (same convention as :meth:`save`).
        """

    def __len__(self) -> int:
        """Return the number of indexed items."""
        return 0
