"""CLIP image and text encoder using open_clip_torch."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import open_clip  # type: ignore[import-untyped]
import torch
from PIL import Image

from vlm_video.common.logging_utils import get_logger

logger = get_logger(__name__)

# Type alias for flexibility
ImageInput = Union[str, Path, np.ndarray, Image.Image]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of *vec*."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


class CLIPEncoder:
    """Encode images and text into L2-normalised embedding vectors using CLIP.

    Parameters
    ----------
    model_name:
        Open CLIP model architecture, e.g. ``"ViT-B-32"``.
    pretrained:
        Pretrained weights tag, e.g. ``"laion2b_s34b_b79k"`` or ``"openai"``.
    device:
        Inference device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model: open_clip.CLIP | None = None
        self._preprocess = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info(
            "Loading CLIP model '%s' pretrained='%s' on %s …",
            self.model_name,
            self.pretrained,
            self.device,
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode_image(self, image_input: ImageInput) -> np.ndarray:
        """Encode a single image to a L2-normalised embedding vector.

        Parameters
        ----------
        image_input:
            A file path (str / Path), a PIL Image, or an RGB numpy array
            (H × W × 3, uint8).

        Returns
        -------
        np.ndarray
            1-D float32 array, L2-normalised.
        """
        self._load()

        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_image = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        tensor = self._preprocess(pil_image).unsqueeze(0).to(self.device)  # type: ignore[misc]
        with torch.no_grad():
            features = self._model.encode_image(tensor)  # type: ignore[union-attr]

        vec = features.cpu().float().numpy().squeeze()
        return _l2_normalize(vec)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text string to a L2-normalised embedding vector.

        Parameters
        ----------
        text:
            Input text (up to the model's context length).

        Returns
        -------
        np.ndarray
            1-D float32 array, L2-normalised.
        """
        self._load()

        tokens = self._tokenizer([text]).to(self.device)  # type: ignore[misc]
        with torch.no_grad():
            features = self._model.encode_text(tokens)  # type: ignore[union-attr]

        vec = features.cpu().float().numpy().squeeze()
        return _l2_normalize(vec)
