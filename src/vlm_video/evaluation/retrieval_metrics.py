"""Information-retrieval evaluation metrics."""

from __future__ import annotations

from typing import Any


def recall_at_k(relevant: set[Any], retrieved: list[Any], k: int) -> float:
    """Compute Recall@k.

    Parameters
    ----------
    relevant:
        Set of relevant item IDs for the query.
    retrieved:
        Ordered list of retrieved item IDs (up to ``k`` are considered).
    k:
        Cutoff rank.

    Returns
    -------
    float
        Recall@k in ``[0, 1]``.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def average_precision(relevant: set[Any], retrieved: list[Any]) -> float:
    """Compute Average Precision for a single query.

    Parameters
    ----------
    relevant:
        Set of relevant item IDs.
    retrieved:
        Ordered list of retrieved item IDs.

    Returns
    -------
    float
        AP in ``[0, 1]``.
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            sum_precision += hits / i

    return sum_precision / len(relevant)


def mean_average_precision(
    relevant_lists: list[set[Any]],
    retrieved_lists: list[list[Any]],
) -> float:
    """Compute Mean Average Precision (MAP) across multiple queries.

    Parameters
    ----------
    relevant_lists:
        List of relevant-item sets, one per query.
    retrieved_lists:
        List of retrieved-item lists, one per query.

    Returns
    -------
    float
        MAP in ``[0, 1]``.
    """
    if not relevant_lists:
        return 0.0
    aps = [
        average_precision(rel, ret)
        for rel, ret in zip(relevant_lists, retrieved_lists)
    ]
    return sum(aps) / len(aps)


def evaluate_retrieval(
    queries_gt: list[dict[str, Any]],
    results: list[dict[str, Any]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Aggregate retrieval metrics across all queries.

    Parameters
    ----------
    queries_gt:
        List of ground-truth dicts.  Each dict must have:

        * ``"query_id"`` – unique query identifier
        * ``"relevant_segment_ids"`` – list/set of relevant segment IDs
    results:
        List of result dicts.  Each dict must have:

        * ``"query_id"`` – query identifier
        * ``"retrieved_segment_ids"`` – ordered list of retrieved segment IDs
    k_values:
        Cutoffs for Recall@k.  Defaults to ``[1, 3, 5]``.

    Returns
    -------
    dict
        ``{"MAP": ..., "Recall@1": ..., "Recall@3": ..., "Recall@5": ...}``
    """
    if k_values is None:
        k_values = [1, 3, 5]

    gt_map: dict[Any, set[Any]] = {
        q["query_id"]: set(q["relevant_segment_ids"]) for q in queries_gt
    }

    relevant_lists: list[set[Any]] = []
    retrieved_lists: list[list[Any]] = []
    recall_sums: dict[int, float] = {k: 0.0 for k in k_values}

    for res in results:
        qid = res["query_id"]
        rel = gt_map.get(qid, set())
        ret = list(res.get("retrieved_segment_ids", []))
        relevant_lists.append(rel)
        retrieved_lists.append(ret)
        for k in k_values:
            recall_sums[k] += recall_at_k(rel, ret, k)

    n = max(len(results), 1)
    metrics: dict[str, float] = {
        "MAP": round(mean_average_precision(relevant_lists, retrieved_lists), 4)
    }
    for k in k_values:
        metrics[f"Recall@{k}"] = round(recall_sums[k] / n, 4)

    return metrics
