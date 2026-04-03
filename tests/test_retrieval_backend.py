"""Unit tests for SklearnIndex retrieval backend."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_video.retrieval.sklearn_index import SklearnIndex


def _make_embeddings(n: int = 5, d: int = 16, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_metadata(n: int = 5) -> list[dict]:
    return [
        {"segment_id": i, "start_time": float(i * 60), "end_time": float((i + 1) * 60)}
        for i in range(n)
    ]


class TestSklearnIndexBuild:
    def test_build_sets_length(self):
        idx = SklearnIndex()
        embs = _make_embeddings(5)
        meta = _make_metadata(5)
        idx.build(embs, meta)
        assert len(idx) == 5

    def test_build_rejects_1d_embeddings(self):
        idx = SklearnIndex()
        with pytest.raises(ValueError, match="2-D"):
            idx.build(np.zeros(16, dtype=np.float32), [{}])

    def test_build_rejects_length_mismatch(self):
        idx = SklearnIndex()
        embs = _make_embeddings(5)
        with pytest.raises(ValueError, match="mismatch"):
            idx.build(embs, _make_metadata(3))

    def test_empty_index_length(self):
        idx = SklearnIndex()
        assert len(idx) == 0


class TestSklearnIndexSearch:
    def setup_method(self):
        self.embs = _make_embeddings(10)
        self.meta = _make_metadata(10)
        self.idx = SklearnIndex()
        self.idx.build(self.embs, self.meta)

    def test_search_returns_top_k(self):
        query = self.embs[0]
        results = self.idx.search(query, top_k=3)
        assert len(results) == 3

    def test_search_first_result_is_query_itself(self):
        # The exact query embedding should match itself with score ≈ 1.0
        query = self.embs[2]
        results = self.idx.search(query, top_k=1)
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
        assert results[0]["segment_id"] == 2

    def test_search_results_sorted_descending(self):
        query = self.embs[0]
        results = self.idx.search(query, top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_metadata_keys(self):
        results = self.idx.search(self.embs[0], top_k=1)
        assert "segment_id" in results[0]
        assert "start_time" in results[0]
        assert "score" in results[0]

    def test_search_empty_index_returns_empty(self):
        idx = SklearnIndex()
        results = idx.search(self.embs[0], top_k=5)
        assert results == []

    def test_top_k_capped_at_index_size(self):
        # Asking for more than available should return all items
        results = self.idx.search(self.embs[0], top_k=100)
        assert len(results) == 10


class TestSklearnIndexSaveLoad:
    def test_save_creates_files(self, tmp_path):
        idx = SklearnIndex()
        idx.build(_make_embeddings(4), _make_metadata(4))
        idx.save(tmp_path)
        assert (tmp_path / "embeddings.npy").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_load_restores_length(self, tmp_path):
        idx = SklearnIndex()
        idx.build(_make_embeddings(4), _make_metadata(4))
        idx.save(tmp_path)

        idx2 = SklearnIndex()
        idx2.load(tmp_path)
        assert len(idx2) == 4

    def test_save_load_search_consistency(self, tmp_path):
        embs = _make_embeddings(6)
        meta = _make_metadata(6)
        idx = SklearnIndex()
        idx.build(embs, meta)
        idx.save(tmp_path)

        idx2 = SklearnIndex()
        idx2.load(tmp_path)

        query = embs[3]
        results1 = idx.search(query, top_k=3)
        results2 = idx2.search(query, top_k=3)

        # Both indexes should return the same top result
        assert results1[0]["segment_id"] == results2[0]["segment_id"]
        assert results1[0]["score"] == pytest.approx(results2[0]["score"], abs=1e-5)

    def test_load_missing_embeddings_raises(self, tmp_path):
        idx = SklearnIndex()
        with pytest.raises(FileNotFoundError, match="embeddings.npy"):
            idx.load(tmp_path)

    def test_load_missing_metadata_raises(self, tmp_path):
        # Create embeddings.npy but not metadata.json
        np.save(tmp_path / "embeddings.npy", _make_embeddings(2))
        idx = SklearnIndex()
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            idx.load(tmp_path)
