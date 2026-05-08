"""Text encoder supporting CLIP or PhoBERT embeddings."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from vlm_video.embeddings.clip_encoder import CLIPEncoder


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


class TextEncoder:
    """Encode text strings using CLIP or PhoBERT.

    Parameters
    ----------
    encoder_type:
        ``"clip"`` or ``"phobert"``.
    clip_encoder:
        Optional pre-instantiated :class:`~vlm_video.embeddings.clip_encoder.CLIPEncoder`.
    clip_model_name:
        Open CLIP model architecture (used if *clip_encoder* is not provided).
    clip_pretrained:
        CLIP pretrained weights tag (used if *clip_encoder* is not provided).
    phobert_model_name:
        HuggingFace model ID for PhoBERT.
    device:
        Inference device.
    """

    def __init__(
        self,
        encoder_type: str = "clip",
        clip_encoder: CLIPEncoder | None = None,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "laion2b_s32b_b82k",
        phobert_model_name: str = "vinai/phobert-base",
        device: str = "cpu",
    ) -> None:
        self.encoder_type = encoder_type.lower()
        self.device = device
        self.phobert_model_name = phobert_model_name
        self._clip_encoder = clip_encoder
        self._phobert_model: AutoModel | None = None
        self._phobert_tokenizer: AutoTokenizer | None = None

        if self.encoder_type == "clip" and self._clip_encoder is None:
            self._clip_encoder = CLIPEncoder(
                model_name=clip_model_name,
                pretrained=clip_pretrained,
                device=device,
            )

    def _load_phobert(self) -> None:
        if self._phobert_model is not None:
            return
        self._phobert_tokenizer = AutoTokenizer.from_pretrained(self.phobert_model_name)
        self._phobert_model = AutoModel.from_pretrained(self.phobert_model_name)
        self._phobert_model.to(self.device)
        self._phobert_model.eval()

    @staticmethod
    def _mean_pool(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], device=hidden.device)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

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
        if self.encoder_type == "clip":
            if self._clip_encoder is None:
                raise RuntimeError("CLIP encoder is not initialised.")
            return self._clip_encoder.encode_text(text)
        if self.encoder_type == "phobert":
            return self._encode_phobert(text)
        raise ValueError(f"Unknown text encoder type: {self.encoder_type!r}")

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
        if self.encoder_type == "clip":
            if self._clip_encoder is None:
                raise RuntimeError("CLIP encoder is not initialised.")
            return np.stack([self._clip_encoder.encode_text(t) for t in texts], axis=0)
        if self.encoder_type == "phobert":
            return self._encode_phobert_batch(texts)
        raise ValueError(f"Unknown text encoder type: {self.encoder_type!r}")

    def _encode_phobert(self, text: str) -> np.ndarray:
        self._load_phobert()
        assert self._phobert_model is not None
        assert self._phobert_tokenizer is not None
        tokens = self._phobert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self._phobert_model(**tokens)
        hidden = outputs.last_hidden_state
        pooled = self._mean_pool(hidden, tokens.get("attention_mask"))
        vec = pooled[0].detach().cpu().float().numpy().astype(np.float32)
        return _l2_normalize(vec)

    def _encode_phobert_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self._load_phobert()
        assert self._phobert_model is not None
        assert self._phobert_tokenizer is not None
        tokens = self._phobert_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self._phobert_model(**tokens)
        hidden = outputs.last_hidden_state
        pooled = self._mean_pool(hidden, tokens.get("attention_mask"))
        vecs = pooled.detach().cpu().float().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return (vecs / norms).astype(np.float32)
