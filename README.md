# Optimizing Segmentation and Event Retrieval in Syllabus Videos using Vision-Language Models (VLM)

> **NCKH Research Project** — Automatic lecture-video segmentation and semantic retrieval using CLIP, Whisper ASR, and optional OCR.

---

## Overview

This repository implements a research pipeline that:

1. **Segments** lecture videos into topically coherent parts using CLIP visual embeddings fused with ASR transcripts (and optionally OCR slide text).
2. **Indexes** segment embeddings for fast semantic retrieval.
3. **Retrieves** the most relevant video segments for a free-text query.
4. **Evaluates** results using boundary F1 and standard IR metrics (MAP, Recall@k).

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Python | ≥ 3.11 | [python.org](https://www.python.org/downloads/) |
| ffmpeg | any | Must be on PATH — see Windows hint below |
| CUDA (optional) | 11.8 / 12.x | CPU works for ViT-B-32 + Whisper base |

### Windows — install ffmpeg

```powershell
winget install ffmpeg          # Windows Package Manager
# OR
choco install ffmpeg           # Chocolatey
```

Then restart your terminal so `ffmpeg` is on PATH.

---

## Windows quickstart (PowerShell)

```powershell
# 1. Clone the repo
git clone https://github.com/anhhoangdn/NCKH
cd NCKH

# 2. Create venv and install
.\run.ps1 setup
.\run.ps1 install

# 3. Activate the environment
.venv\Scripts\Activate.ps1

# 4. Copy and edit environment variables (optional)
Copy-Item .env.example .env
notepad .env
```

## Installation

### Core (CPU, no optional extras)

```bash
pip install -e .
```

### With development tools

```bash
pip install -e ".[dev]"
```

### With OCR support (Tesseract)

```bash
pip install -e ".[ocr]"
# Also install the Tesseract binary:
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux  : sudo apt install tesseract-ocr tesseract-ocr-vie
```

### With FAISS (faster retrieval)

```bash
pip install -e ".[faiss]"
# Note: faiss-cpu on Windows requires WSL or conda:
# conda install -c conda-forge faiss-cpu
```

---

## End-to-end example (single video, CPU)

```bash
# Put your lecture video in data/raw/
# Then run each pipeline step:

python scripts/01_extract_frames.py \
    --input_video data/raw/lec01.mp4 --video_id lec01

python scripts/02_run_asr.py \
    --input_video data/raw/lec01.mp4 --video_id lec01

python scripts/04_build_embeddings.py \
    --frames_dir data/interim/lec01/frames \
    --transcript_jsonl data/interim/lec01/transcript.jsonl \
    --video_id lec01

python scripts/05_segment_video.py \
    --embeddings_npz data/interim/lec01/embeddings.npz \
    --video_id lec01

python scripts/06_build_index.py \
    --segments_jsonl data/interim/lec01/segments_pred.jsonl \
    --video_id lec01

python scripts/07_retrieve.py \
    --index_dir data/interim/lec01/index \
    --query "What is gradient descent?"
```

Or use the single-command pipeline:

```bash
python -m vlm_video.pipelines.run_all data/raw/lec01.mp4 --exp_name lec01
```

---

## Directory structure

```
NCKH/
├── configs/
│   ├── default.yaml                 ← Main configuration file
│   ├── segmentation/
│   │   ├── clip_latefusion.yaml
│   │   └── text_only.yaml
│   └── retrieval/
│       ├── index_default.yaml
│       └── index_faiss.yaml
├── data/
│   ├── raw/          ← Source videos (git-ignored)
│   ├── interim/      ← Pipeline outputs (git-ignored)
│   └── processed/    ← Ground-truth annotations (git-ignored)
├── docs/
│   ├── annotation_guideline.md
│   ├── dataset_format.md
│   └── experiment_protocol.md
├── scripts/
│   ├── 01_extract_frames.py
│   ├── 02_run_asr.py
│   ├── 03_run_ocr.py
│   ├── 04_build_embeddings.py
│   ├── 05_segment_video.py
│   ├── 06_build_index.py
│   ├── 07_retrieve.py
│   └── 08_evaluate.py
├── src/vlm_video/
│   ├── common/          ← Config, JSONL I/O, logging, timestamps
│   ├── preprocess/      ← ffmpeg, Whisper ASR, Tesseract OCR
│   ├── embeddings/      ← CLIP encoder, text encoder, late fusion
│   ├── segmentation/    ← Change scores, thresholding, VideoSegmenter, baselines
│   ├── retrieval/       ← SklearnIndex, FaissIndex, factory, ranking
│   ├── evaluation/      ← Boundary F1, retrieval metrics
│   └── pipelines/       ← EndToEndPipeline
├── tests/
│   ├── test_boundary_f1.py
│   └── test_retrieval_backend.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── run.ps1              ← Windows PowerShell helper
└── README.md
```

---

## Configuration guide

All settings live in `configs/default.yaml`.  Override any value by creating
a custom YAML and passing `--config your_config.yaml` to any script.

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `frame_extraction` | `fps` | `0.5` | Frames per second to extract |
| `asr` | `model` | `base` | Whisper model size |
| `asr` | `language` | `vi` | ISO language code |
| `embeddings` | `model` | `ViT-B-32` | CLIP architecture |
| `embeddings` | `weights.visual` | `0.6` | Visual modality weight |
| `segmentation` | `method` | `clip_latefusion` | Segmentation algorithm |
| `segmentation` | `threshold` | `0.4` | Cosine change threshold |
| `retrieval` | `backend` | `sklearn` | `sklearn` or `faiss` |
| `retrieval` | `top_k` | `5` | Results to return |

See `configs/default.yaml` for all options with inline documentation.

---

## Running tests

```bash
pytest
```

---

## Linting

```bash
ruff check src/ scripts/ tests/
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{nckh2024vlm,
  title  = {Optimizing Segmentation and Event Retrieval in Syllabus Videos using VLMs},
  author = {NCKH Research Team},
  year   = {2024},
  url    = {https://github.com/anhhoangdn/NCKH}
}
```

---

## License

MIT License — see `LICENSE` file for details.
