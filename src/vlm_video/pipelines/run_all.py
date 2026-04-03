"""End-to-end pipeline: frames → ASR → embeddings → segmentation → index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vlm_video.common.config import load_config, resolve_output_dir, validate_config
from vlm_video.common.io_jsonl import read_jsonl, write_jsonl
from vlm_video.common.logging_utils import get_logger
from vlm_video.embeddings.clip_encoder import CLIPEncoder
from vlm_video.embeddings.fusion import late_fusion
from vlm_video.preprocess.asr_wrapper import WhisperASR
from vlm_video.preprocess.ffmpeg_utils import extract_audio, extract_frames
from vlm_video.retrieval.index_factory import get_index
from vlm_video.segmentation.segmenter import VideoSegmenter

logger = get_logger(__name__)


class EndToEndPipeline:
    """Orchestrate the full vlm_video processing pipeline.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file.  Pass *None* to use built-in
        defaults.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.cfg = load_config(config_path)
        validate_config(self.cfg)

    def run(self, video_path: str | Path, exp_name: str | None = None) -> dict[str, Any]:
        """Run the full pipeline on *video_path*.

        Steps
        -----
        1. Extract frames (ffmpeg)
        2. Extract audio and run ASR (faster-whisper)
        3. Build per-frame fused CLIP embeddings
        4. Segment the video
        5. Build a retrieval index over segment embeddings
        6. Save all artefacts to an output directory

        Parameters
        ----------
        video_path:
            Path to the source video file.
        exp_name:
            Optional experiment name used to label the output directory.

        Returns
        -------
        dict
            Summary dict with keys: ``run_dir``, ``n_frames``, ``n_segments``,
            ``index_dir``, ``segments_jsonl``.
        """
        video_path = Path(video_path)
        run_dir = resolve_output_dir(self.cfg, exp_name or video_path.stem)
        logger.info("Run directory: %s", run_dir)

        # ── Step 1: Extract frames ────────────────────────────────────────────
        frames_dir = run_dir / "frames"
        fps = self.cfg["frame_extraction"]["fps"]
        frame_paths = extract_frames(video_path, frames_dir, fps=fps)
        n_frames = len(frame_paths)
        logger.info("Extracted %d frames.", n_frames)

        # Compute frame timestamps from index and fps
        timestamps = [i / fps for i in range(n_frames)]

        # Save frame manifest
        manifest = [
            {"frame_idx": i, "path": p, "timestamp_sec": timestamps[i]}
            for i, p in enumerate(frame_paths)
        ]
        write_jsonl(run_dir / "frame_manifest.jsonl", manifest)

        # ── Step 2: ASR ───────────────────────────────────────────────────────
        asr_cfg = self.cfg["asr"]
        audio_path = run_dir / "audio.wav"
        extract_audio(video_path, audio_path)

        asr = WhisperASR(
            model_size=asr_cfg["model"],
            language=asr_cfg.get("language"),
            device=asr_cfg.get("device", "cpu"),
            compute_type=asr_cfg.get("compute_type", "int8"),
            beam_size=asr_cfg.get("beam_size", 5),
            vad_filter=asr_cfg.get("vad_filter", True),
        )
        transcripts = asr.transcribe(audio_path)
        write_jsonl(run_dir / "transcript.jsonl", transcripts)
        logger.info("ASR produced %d segments.", len(transcripts))

        # ── Step 3: Build embeddings ──────────────────────────────────────────
        emb_cfg = self.cfg["embeddings"]
        encoder = CLIPEncoder(
            model_name=emb_cfg["model"],
            pretrained=emb_cfg["pretrained"],
            device=emb_cfg.get("device", "cpu"),
        )
        w = emb_cfg["weights"]
        wv, wt = w["visual"], w["text"]

        logger.info("Encoding %d frames …", n_frames)
        emb_list: list[np.ndarray] = []
        emb_meta: list[dict[str, Any]] = []

        for i, fp in enumerate(frame_paths):
            t = timestamps[i]
            vis = encoder.encode_image(fp)
            # Find overlapping ASR text
            text = ""
            for seg in transcripts:
                if seg.get("start", 0) <= t <= seg.get("end", 0):
                    text = seg.get("text", "")
                    break
            txt_emb = encoder.encode_text(text) if text else None
            fused = late_fusion(vis, txt_emb, None, wv, wt, 0.0)
            emb_list.append(fused)
            emb_meta.append({"frame_idx": i, "timestamp_sec": t, "asr_text": text})

        embeddings = np.stack(emb_list, axis=0)
        np.savez_compressed(run_dir / "embeddings.npz", embeddings=embeddings)
        write_jsonl(run_dir / "embedding_meta.jsonl", emb_meta)

        # ── Step 4: Segmentation ──────────────────────────────────────────────
        segmenter = VideoSegmenter(self.cfg)
        segments = segmenter.segment(
            frame_paths, timestamps, transcripts=transcripts, embeddings=embeddings
        )
        write_jsonl(run_dir / "segments_pred.jsonl", segments)
        logger.info("Segmentation: %d segments.", len(segments))

        # ── Step 5: Build retrieval index ─────────────────────────────────────
        seg_embeddings = np.array(
            [s["embedding"] for s in segments], dtype=np.float32
        )
        seg_meta = [
            {k: v for k, v in s.items() if k != "embedding"} for s in segments
        ]

        backend = self.cfg["retrieval"]["backend"]
        index = get_index(backend)
        index.build(seg_embeddings, seg_meta)

        index_dir = run_dir / "index"
        index.save(index_dir)
        logger.info("Index saved to %s", index_dir)

        summary = {
            "run_dir": str(run_dir),
            "n_frames": n_frames,
            "n_segments": len(segments),
            "index_dir": str(index_dir),
            "segments_jsonl": str(run_dir / "segments_pred.jsonl"),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Pipeline complete. Summary: %s", summary)
        return summary


def main() -> None:
    """Entry point for the ``vlm-pipeline`` CLI command."""
    import argparse

    parser = argparse.ArgumentParser(description="vlm_video end-to-end pipeline")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--exp_name", default=None, help="Experiment name label")
    args = parser.parse_args()

    pipeline = EndToEndPipeline(config_path=args.config)
    summary = pipeline.run(args.video_path, exp_name=args.exp_name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
