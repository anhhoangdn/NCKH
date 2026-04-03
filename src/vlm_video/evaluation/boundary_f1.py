"""Boundary-detection F1 evaluation metrics."""

from __future__ import annotations


def match_boundaries(
    pred_sec: list[float],
    gt_sec: list[float],
    tolerance: float,
) -> tuple[int, int, int]:
    """Match predicted boundaries to ground-truth boundaries within *tolerance*.

    Each ground-truth boundary may be matched at most once (greedy, sorted by
    distance).  Likewise, each prediction is used at most once.

    Parameters
    ----------
    pred_sec:
        Predicted boundary timestamps in seconds.
    gt_sec:
        Ground-truth boundary timestamps in seconds.
    tolerance:
        Maximum absolute time difference (seconds) for a match to count as TP.

    Returns
    -------
    tuple[int, int, int]
        ``(tp, fp, fn)`` counts.
    """
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    # Sort predictions by timestamp for deterministic greedy matching
    for pi, p in enumerate(sorted(pred_sec)):
        best_gi = -1
        best_dist = float("inf")
        for gi, g in enumerate(gt_sec):
            if gi in matched_gt:
                continue
            dist = abs(p - g)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_gi = gi
        if best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    tp = len(matched_gt)
    fp = len(pred_sec) - tp
    fn = len(gt_sec) - tp
    return tp, fp, fn


def boundary_f1(
    pred_sec: list[float],
    gt_sec: list[float],
    tolerances: list[float] | None = None,
) -> dict[float, dict[str, float]]:
    """Compute Precision, Recall, and F1 at multiple tolerances.

    Parameters
    ----------
    pred_sec:
        Predicted boundary timestamps in seconds.
    gt_sec:
        Ground-truth boundary timestamps in seconds.
    tolerances:
        List of tolerance values in seconds.  Defaults to ``[5, 10]``.

    Returns
    -------
    dict
        Mapping of ``{tolerance: {"P": ..., "R": ..., "F1": ...}}``.

    Examples
    --------
    >>> boundary_f1([10.0, 60.0], [12.0, 58.0], tolerances=[5])
    {5: {'P': 1.0, 'R': 1.0, 'F1': 1.0}}
    """
    if tolerances is None:
        tolerances = [5.0, 10.0]

    results: dict[float, dict[str, float]] = {}

    for tol in tolerances:
        tp, fp, fn = match_boundaries(pred_sec, gt_sec, tol)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results[tol] = {"P": round(precision, 4), "R": round(recall, 4), "F1": round(f1, 4)}

    return results
