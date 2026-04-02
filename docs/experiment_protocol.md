# Experiment Protocol

Step-by-step instructions for running a full experiment with vlm_video.

---

## Prerequisites

- Python 3.11+
- ffmpeg installed and on PATH  
  Windows: `winget install ffmpeg`  
  Linux: `sudo apt install ffmpeg`
- (Optional) Tesseract OCR for slide text extraction
- GPU recommended for large models; CPU works for `ViT-B-32` + `base` Whisper

---

## 1. Environment setup

### Windows (PowerShell)

```powershell
.\run.ps1 setup
.\run.ps1 install
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 2. Place raw data

Put your lecture videos in `data/raw/`:

```
data/raw/
├── lec01.mp4
├── lec02.mp4
└── ...
```

Use short, ASCII-only identifiers (no spaces) as video IDs.

---

## 3. Run the pipeline (step by step)

Replace `lec01` with your actual `--video_id`.

### Step 1 — Extract frames

```bash
python scripts/01_extract_frames.py \
    --input_video data/raw/lec01.mp4 \
    --video_id lec01 \
    --config configs/default.yaml
```

### Step 2 — Run ASR

```bash
python scripts/02_run_asr.py \
    --input_video data/raw/lec01.mp4 \
    --video_id lec01 \
    --config configs/default.yaml
```

### Step 3 — (Optional) Run OCR

Enable OCR in config first: `ocr.enabled: true`

```bash
python scripts/03_run_ocr.py \
    --frames_dir data/interim/lec01/frames \
    --video_id lec01 \
    --config configs/default.yaml
```

### Step 4 — Build embeddings

```bash
python scripts/04_build_embeddings.py \
    --frames_dir data/interim/lec01/frames \
    --transcript_jsonl data/interim/lec01/transcript.jsonl \
    --video_id lec01 \
    --config configs/default.yaml
```

### Step 5 — Segment the video

```bash
python scripts/05_segment_video.py \
    --embeddings_npz data/interim/lec01/embeddings.npz \
    --video_id lec01 \
    --config configs/default.yaml
```

### Step 6 — Build retrieval index

```bash
python scripts/06_build_index.py \
    --segments_jsonl data/interim/lec01/segments_pred.jsonl \
    --video_id lec01 \
    --config configs/default.yaml
```

### Step 7 — Query the index

```bash
python scripts/07_retrieve.py \
    --index_dir data/interim/lec01/index \
    --query "What is backpropagation?" \
    --top_k 5
```

### Step 8 — Evaluate (requires annotations)

```bash
python scripts/08_evaluate.py \
    --pred_jsonl data/interim/lec01/segments_pred.jsonl \
    --gt_jsonl data/processed/my_dataset/annotations.jsonl \
    --out_dir outputs/eval/lec01
```

---

## 4. End-to-end pipeline (single command)

```bash
python -m vlm_video.pipelines.run_all data/raw/lec01.mp4 \
    --config configs/default.yaml \
    --exp_name lec01_baseline
```

---

## 5. Comparing methods

To compare `clip_latefusion` vs `text_only`:

```bash
# Baseline: text-only
python scripts/05_segment_video.py \
    --embeddings_npz data/interim/lec01/embeddings.npz \
    --video_id lec01 \
    --config configs/segmentation/text_only.yaml

# Main: CLIP late fusion (default)
python scripts/05_segment_video.py \
    --embeddings_npz data/interim/lec01/embeddings.npz \
    --video_id lec01 \
    --config configs/segmentation/clip_latefusion.yaml
```

Then evaluate both with `scripts/08_evaluate.py` pointing to each output.

---

## 6. Logging

Set the `LOG_LEVEL` environment variable to control verbosity:

```bash
# Linux / macOS
export LOG_LEVEL=DEBUG

# Windows PowerShell
$env:LOG_LEVEL = "DEBUG"
```

Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

---

## 7. Reproducibility checklist

- [ ] Commit the exact `configs/` files used
- [ ] Record model names and pretrained tags in your experiment notes
- [ ] Save `outputs/runs/<run_dir>/summary.json`
- [ ] Archive `data/interim/<video_id>/` for each experiment
- [ ] Document software versions: `pip freeze > requirements_lock.txt`
