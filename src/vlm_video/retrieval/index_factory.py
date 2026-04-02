"""Factory for retrieval index backends."""

from __future__ import annotations

import warnings

from vlm_video.retrieval.base_index import BaseIndex


def get_index(backend: str = "sklearn") -> BaseIndex:
    """Instantiate and return the requested retrieval index.

    Parameters
    ----------
    backend:
        ``"sklearn"`` (default) or ``"faiss"``.  If ``"faiss"`` is requested
        but ``faiss`` is not installed, falls back to ``SklearnIndex`` with
        a warning.

    Returns
    -------
    BaseIndex
        A concrete retrieval index instance (not yet populated — call
        :meth:`~vlm_video.retrieval.base_index.BaseIndex.build` next).
    """
    if backend == "faiss":
        try:
            from vlm_video.retrieval.faiss_index import FaissIndex

            return FaissIndex()
        except ImportError as exc:
            warnings.warn(
                f"FAISS import failed ({exc}). "
                "Falling back to SklearnIndex. "
                "Install FAISS with: pip install vlm_video[faiss]",
                stacklevel=2,
            )
            from vlm_video.retrieval.sklearn_index import SklearnIndex

            return SklearnIndex()

    if backend == "sklearn":
        from vlm_video.retrieval.sklearn_index import SklearnIndex

        return SklearnIndex()

    raise ValueError(
        f"Unknown retrieval backend: {backend!r}. "
        "Valid options are 'sklearn' and 'faiss'."
    )
