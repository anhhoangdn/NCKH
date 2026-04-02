"""Optional OCR wrapper using pytesseract.

Install the optional dependency with::

    pip install vlm_video[ocr]
"""

from __future__ import annotations

from pathlib import Path

try:
    import pytesseract  # type: ignore[import-untyped]
    from PIL import Image
except ImportError as _err:
    raise ImportError(
        "pytesseract is required for OCR but is not installed.\n"
        "Install pytesseract: pip install vlm_video[ocr]\n"
        "You also need the Tesseract-OCR binary:\n"
        "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "  Linux  : sudo apt install tesseract-ocr tesseract-ocr-vie\n"
    ) from _err

from vlm_video.common.logging_utils import get_logger

logger = get_logger(__name__)


class TesseractOCR:
    """Extract text from images using Tesseract via pytesseract.

    Parameters
    ----------
    lang:
        Tesseract language string (e.g. ``"vie+eng"``).
    psm:
        Page Segmentation Mode (0–13).  Default ``3`` (fully automatic).
    tesseract_cmd:
        Path to the Tesseract binary.  If *None*, relies on PATH.
    """

    def __init__(
        self,
        lang: str = "vie+eng",
        psm: int = 3,
        tesseract_cmd: str | None = None,
    ) -> None:
        self.lang = lang
        self.psm = psm
        if tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, image_path: str | Path) -> str:
        """Run OCR on *image_path* and return the recognised text.

        Parameters
        ----------
        image_path:
            Path to an image file (JPEG, PNG, …).

        Returns
        -------
        str
            Extracted text, stripped of leading/trailing whitespace.
            Returns an empty string if no text is detected.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        config = f"--psm {self.psm}"
        try:
            image = Image.open(image_path)
            text: str = pytesseract.image_to_string(image, lang=self.lang, config=config)
            return text.strip()
        except Exception as exc:
            logger.warning("OCR failed for %s: %s", image_path.name, exc)
            return ""
