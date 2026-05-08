"""Query encoding utilities for retrieval."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sentence_transformers import SentenceTransformer

_OOP_EQUIVALENTS: Dict[str, str] = {
    "encapsulation": "encapsulation đóng gói",
    "inheritance": "inheritance kế thừa",
    "polymorphism": "polymorphism đa hình",
    "abstraction": "abstraction trừu tượng",
}


def expand_query(query: str) -> str:
    """Expand *query* with Vietnamese equivalents for common OOP terms."""
    tokens = query.split()
    expansions = [
        _OOP_EQUIVALENTS[token.lower()]
        for token in tokens
        if token.lower() in _OOP_EQUIVALENTS
    ]
    if not expansions:
        return query
    return " ".join([query, *expansions]).strip()


def format_query(query: str) -> str:
    """Prefix query text for E5 models."""
    return f"query: {query}"


def format_passage(passage: str) -> str:
    """Prefix passage text for E5 models."""
    return f"passage: {passage}"


class QueryEncoder:
    """SentenceTransformer-based query encoder for multilingual E5."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def encode(self, query: str) -> np.ndarray:
        """Encode *query* into a vector using E5 formatting."""
        self._load()
        assert self._model is not None
        return self._model.encode(format_query(query))
