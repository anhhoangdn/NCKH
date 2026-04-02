"""Convenience text encoder wrapping CLIPEncoder.encode_text."""

from __future__ import annotations

import numpy as np

from vlm_video.embeddings.clip_encoder import CLIPEncoder


class TextEncoder:
    """Encode text strings to L2-normalised CLIP embedding vectors.

    This is a lightweight wrapper around :class:`~vlm_video.embeddings.clip_encoder.CLIPEncoder`
    that exposes only the text-encoding functionality.

    Parameters
    ----------
    model_name:
        Open CLIP model architecture.
    pretrained:
        Pretrained weights tag.
    device:
        Inference device.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        self._encoder = CLIPEncoder(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )

    def encode(self, text: str) -> np.ndarray:
        """Encode *text* to a 1-D float32 L2-normalised vector.

        Parameters
        ----------
        text:
            Input text string.

        Returns
        -------
        np.ndarray
            Embedding vector.
        """
        return self._encoder.encode_text(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings, returning a 2-D array (N × D).

        Parameters
        ----------
        texts:
            List of text strings.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(texts), embedding_dim)``.
        """
        return np.stack([self._encoder.encode_text(t) for t in texts], axis=0)
