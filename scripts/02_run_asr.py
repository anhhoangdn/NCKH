"""Script 02 — Extract audio and run Whisper ASR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vlm_video.common.io_jsonl import write_jsonl
from vlm_video.common.logging_utils import get_logger
from vlm_video.preprocess.asr_wrapper import WhisperASR
from vlm_video.preprocess.ffmpeg_utils import check_ffmpeg, extract_audio

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract audio from a video and transcribe with Whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--input_video", required=True, help="Path to the source video")
    p.add_argument(
        "--out_dir",
        default="data/interim",
        help="Base output directory; outputs go to <out_dir>/<video_id>/",
    )
    p.add_argument("--video_id", required=True, help="Identifier for this video")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from vlm_video.common.config import load_config

    cfg = load_config(args.config)
    asr_cfg = cfg["asr"]

    if not check_ffmpeg():
        sys.exit(1)

    video_path = Path(args.input_video)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    vid_dir = Path(args.out_dir) / args.video_id
    vid_dir.mkdir(parents=True, exist_ok=True)

    # Extract audio
    audio_path = vid_dir / "audio.wav"
    extract_audio(video_path, audio_path)

    # Run ASR
    asr = WhisperASR(
        model_size=asr_cfg["model"],
        language=asr_cfg.get("language"),
        device=asr_cfg.get("device", "cpu"),
        compute_type=asr_cfg.get("compute_type", "int8"),
        beam_size=asr_cfg.get("beam_size", 5),
        vad_filter=asr_cfg.get("vad_filter", True),
    )
    transcripts = asr.transcribe(audio_path)

    # Add video_id to each record
    for seg in transcripts:
        seg["video_id"] = args.video_id

    transcript_path = vid_dir / "transcript.jsonl"
    n = write_jsonl(transcript_path, transcripts)
    logger.info("Saved %d transcript segments to %s", n, transcript_path)
    print(f"Transcribed {n} segments → {transcript_path}")


if __name__ == "__main__":
    main()
