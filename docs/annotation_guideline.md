# Annotation Guideline

## Purpose

This document describes how to manually annotate video boundaries and
retrieval relevance for the vlm_video research project.

---

## 1. Boundary annotation

### What is a segment boundary?

A segment boundary marks the point in time where the topic or content of a
lecture video changes significantly.  Examples include:

- Instructor moving from one chapter to the next
- New concept introduced that is clearly distinct from the previous one
- Slide transition from e.g. "Introduction" to "Methodology"
- Return from a digression or Q&A back to the main content

### What is NOT a boundary?

- Minor pauses or filler words
- Repetition or summary of the immediately preceding content
- Camera cuts or speaker changes that do not alter the topic

### Annotation procedure

1. Watch the video in full at least once before annotating.
2. Re-watch at half speed and note candidate boundary timestamps.
3. For each candidate, record the timestamp (in seconds or HH:MM:SS) at
   which the new topic **begins** (not where the previous one ends).
4. Verify that each boundary satisfies the minimum duration rule
   (≥ 5 seconds between consecutive boundaries; see `configs/default.yaml`).
5. Write each annotated video to the ground-truth JSONL format (see below).

### JSONL format for boundaries

```json
{"video_id": "lec01", "boundaries_sec": [120.0, 340.5, 600.0], "annotator": "A1", "notes": ""}
```

Field descriptions:

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | str | Unique identifier matching `data/raw/<video_id>` |
| `boundaries_sec` | list[float] | Sorted list of boundary timestamps in seconds |
| `annotator` | str | Annotator initials or ID |
| `notes` | str | Optional free-text comments |

---

## 2. Retrieval relevance annotation

### Task description

For each evaluation query, annotators must judge which video segments are
relevant to the query.

### Relevance scale (binary)

| Label | Meaning |
|-------|---------|
| 1 | The segment contains content that directly answers or relates to the query |
| 0 | The segment is not relevant |

### Procedure

1. Read the query text carefully.
2. Watch (or read the transcript of) each candidate segment.
3. Mark it as relevant (1) if the segment content would help a student who
   asked this question.
4. Annotation is per-segment; multiple segments can be relevant to the same query.

### JSONL format for retrieval ground truth

```json
{"query_id": "q001", "query_text": "What is backpropagation?", "video_id": "lec01",
 "relevant_segment_ids": [3, 7, 8]}
```

Field descriptions:

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | str | Unique query identifier |
| `query_text` | str | The natural-language query |
| `video_id` | str | Source video (may be `null` for cross-video queries) |
| `relevant_segment_ids` | list[int] | Segment indices that are relevant |

---

## 3. Inter-annotator agreement

- Each video should be annotated by at least **2 independent annotators**.
- For boundaries: compute temporal agreement within ±5 s tolerance.
- For retrieval: compute Cohen's Kappa on binary relevance labels.
- Disagreements should be resolved by a third annotator or by discussion.

---

## 4. File naming conventions

Store annotations in `data/processed/<dataset_name>/`:

```
annotations.jsonl    ← Boundary annotations
queries.jsonl        ← Retrieval queries with relevance labels
```
