"""Late-fusion of visual, text, and OCR embeddings."""

from __future__ import annotations

import numpy as np


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


def late_fusion(
    visual_emb: np.ndarray | None,
    text_emb: np.ndarray | None,
    ocr_emb: np.ndarray | None = None,
    wv: float = 0.6,
    wt: float = 0.3,
    wo: float = 0.1,
) -> np.ndarray:
    """Compute a weighted late-fusion of up to three embedding modalities.

    Any modality whose embedding is *None* is skipped; the remaining weights
    are renormalised to sum to 1.0 before combining.

    Parameters
    ----------
    visual_emb:
        Visual (CLIP image) embedding, shape ``(D,)``.  Pass *None* to skip.
    text_emb:
        Text (CLIP / ASR) embedding, shape ``(D,)``.  Pass *None* to skip.
    ocr_emb:
        OCR text embedding, shape ``(D,)``.  Pass *None* to skip.
    wv:
        Base weight for the visual modality.
    wt:
        Base weight for the text modality.
    wo:
        Base weight for the OCR modality.

    Returns
    -------
    np.ndarray
        L2-normalised fused embedding of shape ``(D,)``.

    Raises
    ------
    ValueError
        If all three modalities are *None*.
    """
    pairs: list[tuple[np.ndarray, float]] = []
    for emb, w in [(visual_emb, wv), (text_emb, wt), (ocr_emb, wo)]:
        if emb is not None:
            pairs.append((emb, w))

    if not pairs:
        raise ValueError("At least one embedding modality must be non-None.")

    # Renormalise weights so they sum to 1
    total_w = sum(w for _, w in pairs)
    if total_w < 1e-10:
        total_w = 1.0

    fused = sum((emb * (w / total_w)) for emb, w in pairs)
    return _l2_normalize(fused)  # type: ignore[arg-type]
