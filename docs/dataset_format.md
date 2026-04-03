# Dataset Format Reference

All intermediate and final data files use **JSONL** (newline-delimited JSON).
Each line is a self-contained JSON object.  This makes the files easy to
stream, inspect with `jq`, and load incrementally.

---

## Frame manifest (`frame_manifest.jsonl`)

Produced by `scripts/01_extract_frames.py`.

```json
{"video_id": "lec01", "frame_idx": 0, "path": "data/interim/lec01/frames/frame_000001.jpg", "timestamp_sec": 0.0}
{"video_id": "lec01", "frame_idx": 1, "path": "data/interim/lec01/frames/frame_000002.jpg", "timestamp_sec": 2.0}
```

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | str | Video identifier |
| `frame_idx` | int | Zero-based frame index |
| `path` | str | Absolute or relative path to the JPEG file |
| `timestamp_sec` | float | Frame timestamp in seconds |

---

## Transcript (`transcript.jsonl`)

Produced by `scripts/02_run_asr.py` (faster-whisper output).

```json
{"video_id": "lec01", "start": 0.0, "end": 4.5, "text": "Xin chào các bạn, hôm nay chúng ta học..."}
{"video_id": "lec01", "start": 4.5, "end": 9.2, "text": "Bài học đầu tiên là về mạng nơ-ron."}
```

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | str | Video identifier |
| `start` | float | Segment start time in seconds |
| `end` | float | Segment end time in seconds |
| `text` | str | Transcribed text for this time window |

---

## OCR results (`ocr_results.jsonl`)

Produced by `scripts/03_run_ocr.py` (optional).

```json
{"video_id": "lec01", "frame_idx": 0, "frame_path": "...", "ocr_text": "Deep Learning\nChapter 1"}
```

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | str | Video identifier |
| `frame_idx` | int | Frame index |
| `frame_path` | str | Path to the corresponding frame |
| `ocr_text` | str | Extracted text (empty string if none detected) |

---

## Embedding metadata (`embedding_meta.jsonl`)

Produced by `scripts/04_build_embeddings.py`.
The actual embedding vectors are stored in the companion `embeddings.npz`.

```json
{"video_id": "lec01", "frame_idx": 0, "frame_path": "...", "timestamp_sec": 0.0, "asr_text": "...", "ocr_text": ""}
```

The `embeddings.npz` file contains one key `"embeddings"` with shape `(N, D)`
where N = number of frames and D = CLIP embedding dimension (e.g. 512 for ViT-B-32).

---

## Predicted segments (`segments_pred.jsonl`)

Produced by `scripts/05_segment_video.py`.

```json
{
  "video_id": "lec01",
  "start_time": 0.0,
  "end_time": 118.5,
  "frame_indices": [0, 1, 2, ..., 59],
  "embedding": [0.023, -0.11, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | str | Video identifier |
| `start_time` | float | Segment start in seconds |
| `end_time` | float | Segment end in seconds |
| `frame_indices` | list[int] | Frame indices belonging to this segment |
| `embedding` | list[float] | Mean-pooled L2-normalised segment embedding |

---

## Retrieval results (`retrieval_results.jsonl`)

Produced by `scripts/07_retrieve.py`.

```json
{"query": "What is gradient descent?", "rank": 1, "score": 0.912, "video_id": "lec01", "start_time": 340.5, "end_time": 480.0, "segment_id": 3}
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | str | Original query text |
| `rank` | int | Result rank (1 = best) |
| `score` | float | Cosine similarity score |
| `video_id` | str | Source video |
| `start_time` | float | Segment start in seconds |
| `end_time` | float | Segment end in seconds |
| `segment_id` | int | Segment index within the video |

---

## Ground-truth annotations

See [`annotation_guideline.md`](annotation_guideline.md) for the formats of
`annotations.jsonl` and `queries.jsonl`.
