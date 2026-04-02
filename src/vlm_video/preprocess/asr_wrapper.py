"""Automatic Speech Recognition wrapper using faster-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_video.common.logging_utils import get_logger

logger = get_logger(__name__)


class WhisperASR:
    """Thin wrapper around :mod:`faster_whisper` for audio transcription.

    Parameters
    ----------
    model_size:
        Whisper model size: ``"tiny"``, ``"base"``, ``"small"``, ``"medium"``,
        ``"large-v2"``, or ``"large-v3"``.
    language:
        ISO-639-1 language code (e.g. ``"vi"`` for Vietnamese, ``"en"`` for
        English).  Pass ``None`` for automatic detection.
    device:
        Inference device: ``"cpu"`` or ``"cuda"``.
    compute_type:
        Quantisation type: ``"int8"``, ``"float16"``, ``"float32"``.
    beam_size:
        Beam search width.
    vad_filter:
        Whether to apply Voice Activity Detection filtering.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str | None = "vi",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        logger.info(
            "Loading Whisper model '%s' on %s (%s)…",
            self.model_size,
            self.device,
            self.compute_type,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: str | Path) -> list[dict[str, Any]]:
        """Transcribe *audio_path* and return a list of word/segment dicts.

        Parameters
        ----------
        audio_path:
            Path to a WAV (or any format supported by faster-whisper) file.

        Returns
        -------
        list[dict]
            Each dict contains:

            * ``start`` – segment start time in seconds (float)
            * ``end``   – segment end time in seconds (float)
            * ``text``  – transcribed text (str)
        """
        self._load_model()
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing %s …", audio_path.name)
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        results: list[dict[str, Any]] = []
        for seg in segments:
            results.append(
                {
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": seg.text.strip(),
                }
            )

        logger.info("Transcription complete: %d segments.", len(results))
        return results
