"""Script 08 — Evaluate segmentation and retrieval results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm_video.common.io_jsonl import read_jsonl
from vlm_video.common.logging_utils import get_logger
from vlm_video.evaluation.boundary_f1 import boundary_f1
from vlm_video.evaluation.retrieval_metrics import evaluate_retrieval

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate segmentation (boundary F1) and retrieval metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument(
        "--pred_jsonl",
        default=None,
        help="Path to predicted segments JSONL (segments_pred.jsonl)",
    )
    p.add_argument(
        "--gt_jsonl",
        default=None,
        help="Path to ground-truth annotations JSONL",
    )
    p.add_argument(
        "--retrieval_results_jsonl",
        default=None,
        help="Path to retrieval results JSONL (from script 07)",
    )
    p.add_argument(
        "--retrieval_gt_jsonl",
        default=None,
        help="Path to ground-truth retrieval relevance JSONL",
    )
    p.add_argument(
        "--out_dir",
        default="outputs/eval",
        help="Directory where metrics.json will be saved",
    )
    return p.parse_args()


def eval_segmentation(
    pred_path: Path,
    gt_path: Path,
    tolerances: list[float],
) -> dict[str, Any]:
    """Run boundary F1 evaluation."""
    pred_records = list(read_jsonl(pred_path))
    gt_records = list(read_jsonl(gt_path))

    # Group by video_id
    pred_by_vid: dict[str, list[float]] = {}
    for r in pred_records:
        vid = r.get("video_id", "default")
        pred_by_vid.setdefault(vid, []).append(r.get("start_time", 0.0))

    gt_by_vid: dict[str, list[float]] = {}
    for r in gt_records:
        vid = r.get("video_id", "default")
        for b in r.get("boundaries_sec", []):
            gt_by_vid.setdefault(vid, []).append(float(b))

    all_metrics: dict[str, Any] = {}
    for vid in set(pred_by_vid) | set(gt_by_vid):
        pred_sec = pred_by_vid.get(vid, [])
        gt_sec = gt_by_vid.get(vid, [])
        all_metrics[vid] = boundary_f1(pred_sec, gt_sec, tolerances=tolerances)

    # Macro-average across videos
    avg: dict[float, dict[str, float]] = {}
    for tol in tolerances:
        all_p = [all_metrics[v][tol]["P"] for v in all_metrics if tol in all_metrics[v]]
        all_r = [all_metrics[v][tol]["R"] for v in all_metrics if tol in all_metrics[v]]
        all_f = [all_metrics[v][tol]["F1"] for v in all_metrics if tol in all_metrics[v]]
        avg[tol] = {
            "P": round(sum(all_p) / max(len(all_p), 1), 4),
            "R": round(sum(all_r) / max(len(all_r), 1), 4),
            "F1": round(sum(all_f) / max(len(all_f), 1), 4),
        }

    return {"per_video": all_metrics, "macro_avg": avg}


def main() -> None:
    args = parse_args()

    from vlm_video.common.config import load_config

    cfg = load_config(args.config)
    tolerances = [float(t) for t in cfg["evaluation"]["tolerance_sec"]]
    k_values = cfg["evaluation"].get("retrieval_k_values", [1, 3, 5])

    metrics: dict[str, Any] = {}

    # ── Segmentation evaluation ───────────────────────────────────────────────
    if args.pred_jsonl and args.gt_jsonl:
        pred_path = Path(args.pred_jsonl)
        gt_path = Path(args.gt_jsonl)
        if pred_path.exists() and gt_path.exists():
            logger.info("Evaluating segmentation …")
            metrics["segmentation"] = eval_segmentation(pred_path, gt_path, tolerances)
            print("\nSegmentation (macro avg):")
            for tol, m in metrics["segmentation"]["macro_avg"].items():
                print(f"  tol={tol}s  P={m['P']:.4f}  R={m['R']:.4f}  F1={m['F1']:.4f}")
        else:
            logger.warning("Skipping segmentation eval: files not found.")

    # ── Retrieval evaluation ──────────────────────────────────────────────────
    if args.retrieval_results_jsonl and args.retrieval_gt_jsonl:
        ret_path = Path(args.retrieval_results_jsonl)
        ret_gt_path = Path(args.retrieval_gt_jsonl)
        if ret_path.exists() and ret_gt_path.exists():
            logger.info("Evaluating retrieval …")
            results = list(read_jsonl(ret_path))
            queries_gt = list(read_jsonl(ret_gt_path))
            metrics["retrieval"] = evaluate_retrieval(
                queries_gt, results, k_values=k_values
            )
            print("\nRetrieval metrics:")
            for k, v in metrics["retrieval"].items():
                print(f"  {k}: {v:.4f}")
        else:
            logger.warning("Skipping retrieval eval: files not found.")

    if not metrics:
        print("No evaluation performed. Provide --pred_jsonl/--gt_jsonl or retrieval files.")
        return

    # Save metrics
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
