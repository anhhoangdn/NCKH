"""Configuration loading, validation, and output-directory utilities."""

from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ── Defaults (lowest priority) ────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "video": {"supported_extensions": [".mp4", ".avi", ".mkv", ".mov", ".webm"]},
    "frame_extraction": {"fps": 0.5, "format": "jpg", "quality": 90, "max_dim": 512},
    "asr": {
        "model": "base",
        "language": "vi",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 5,
        "vad_filter": True,
    },
    "ocr": {"enabled": False, "lang": "vie+eng", "psm": 3},
    "embeddings": {
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "device": "cpu",
        "batch_size": 32,
        "weights": {"visual": 0.6, "text": 0.3, "ocr": 0.1},
    },
    "segmentation": {
        "method": "clip_latefusion",
        "threshold": 0.4,
        "adaptive_percentile": 85,
        "min_duration": 5,
        "min_segment_duration": 30,
        "smooth_window": 3,
        "merge_sim_threshold": 0.9,
    },
    "retrieval": {"backend": "sklearn", "top_k": 5, "metric": "cosine"},
    "evaluation": {"tolerance_sec": [5, 10], "retrieval_k_values": [1, 3, 5]},
    "output": {
        "base_dir": "outputs/runs",
        "save_embeddings": True,
        "save_segments": True,
        "save_index": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file and deep-merge it with built-in defaults.

    Parameters
    ----------
    path:
        Path to a YAML configuration file.  Pass *None* to use only defaults.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    cfg = copy.deepcopy(_DEFAULTS)
    if path is None:
        return cfg

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}

    return _deep_merge(cfg, user_cfg)


def validate_config(cfg: dict[str, Any]) -> None:
    """Run basic sanity checks on *cfg*.  Raises *ValueError* on problems."""
    # Named tolerance bounds for embedding-weight sum validation
    WEIGHT_SUM_MIN: float = 0.99
    WEIGHT_SUM_MAX: float = 1.01

    fps = cfg.get("frame_extraction", {}).get("fps", 0)
    if fps <= 0:
        raise ValueError(f"frame_extraction.fps must be > 0, got {fps}")

    weights = cfg.get("embeddings", {}).get("weights", {})
    total = sum(weights.get(k, 0) for k in ("visual", "text", "ocr"))
    if not (WEIGHT_SUM_MIN <= total <= WEIGHT_SUM_MAX):
        raise ValueError(
            f"embeddings.weights must sum to ~1.0, got {total:.3f}. "
            "Adjust visual/text/ocr weights in your config."
        )

    top_k = cfg.get("retrieval", {}).get("top_k", 0)
    if top_k < 1:
        raise ValueError(f"retrieval.top_k must be >= 1, got {top_k}")

    threshold = cfg.get("segmentation", {}).get("threshold", -1)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"segmentation.threshold must be in [0, 1], got {threshold}")


def resolve_output_dir(cfg: dict[str, Any], exp_name: str | None = None) -> Path:
    """Create and return a timestamped run directory under *output.base_dir*.

    Parameters
    ----------
    cfg:
        Merged configuration dictionary.
    exp_name:
        Optional experiment name suffix.  Defaults to ``"exp"``.

    Returns
    -------
    Path
        Path to the newly created run directory.
    """
    base = Path(cfg.get("output", {}).get("base_dir", "outputs/runs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = exp_name or "exp"
    run_dir = base / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
