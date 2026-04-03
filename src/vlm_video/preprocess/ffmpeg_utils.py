"""FFmpeg-based frame and audio extraction utilities."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from vlm_video.common.logging_utils import get_logger

logger = get_logger(__name__)


def check_ffmpeg() -> bool:
    """Check whether ``ffmpeg`` is available on PATH.

    Returns
    -------
    bool
        ``True`` if ffmpeg is found, ``False`` otherwise.  Prints a
        Windows-friendly installation hint when not found.
    """
    if shutil.which("ffmpeg") is not None:
        return True

    logger.warning("ffmpeg not found on PATH.")
    print(
        "\n[vlm_video] ffmpeg is required but was not found.\n"
        "Windows install options:\n"
        "  winget : winget install ffmpeg\n"
        "  choco  : choco install ffmpeg\n"
        "  manual : https://www.gyan.dev/ffmpeg/builds/\n"
        "After installing, add ffmpeg.exe to your PATH and restart your terminal.\n"
    )
    return False


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float = 0.5,
) -> list[str]:
    """Extract frames from *video_path* at *fps* frames-per-second.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    out_dir:
        Directory where JPEG frames will be written.
    fps:
        Extraction rate in frames per second (default: 0.5).

    Returns
    -------
    list[str]
        Sorted list of absolute paths to the extracted frame files.

    Raises
    ------
    RuntimeError
        If ffmpeg is not found or the extraction command fails.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg not found. See install hint above.")

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]

    logger.info("Extracting frames: %s  fps=%s  →  %s", video_path.name, fps, out_dir)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ffmpeg stderr:\n%s", result.stderr)
        raise RuntimeError(f"Frame extraction failed (exit code {result.returncode})")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    logger.info("Extracted %d frames.", len(frames))
    return [str(f) for f in frames]


def extract_audio(
    video_path: str | Path,
    out_path: str | Path,
) -> str:
    """Extract audio from *video_path* to a mono 16 kHz WAV file.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    out_path:
        Destination WAV file path.

    Returns
    -------
    str
        Absolute path to the extracted audio file.

    Raises
    ------
    RuntimeError
        If ffmpeg is not found or the extraction command fails.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg not found. See install hint above.")

    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_path),
    ]

    logger.info("Extracting audio: %s  →  %s", video_path.name, out_path)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ffmpeg stderr:\n%s", result.stderr)
        raise RuntimeError(f"Audio extraction failed (exit code {result.returncode})")

    logger.info("Audio written to %s", out_path)
    return str(out_path)
