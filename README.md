# Optimizing Segmentation and Event Retrieval in Syllabus Videos using Vision-Language Models (VLM)

> **Research Project** — Automatic lecture-video segmentation and semantic retrieval using CLIP, Whisper ASR, and optional OCR.

---

## Overview

This repository implements a research pipeline that:

1. **Segments** lecture videos into topically coherent parts using CLIP visual embeddings fused with ASR transcripts (and optionally OCR slide text).
2. **Indexes** segment embeddings for fast semantic retrieval.
3. **Retrieves** the most relevant video segments for a free-text query, using hybrid semantic + keyword scoring and optional LLM reranking via Claude.
4. **Evaluates** results using boundary F1 and standard IR metrics (MAP, Recall@k).

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Python | ≥ 3.11 | [python.org](https://www.python.org/downloads/) |
| ffmpeg | any | Must be on PATH — see Windows hint below |
| CUDA (optional) | 11.8 / 12.x | CPU works for ViT-L-14 + Whisper large-v3 (slower) |

### Windows — install ffmpeg

```cmd
winget install ffmpeg          :: Windows Package Manager
:: OR
choco install ffmpeg           :: Chocolatey
```

Then restart your terminal so `ffmpeg` is on PATH.

---

## Windows quickstart

```cmd
:: 1. Clone the repo
git clone https://github.com/anhhoangdn/vlm-video-segmentation
cd vlm-video-segmentation

:: 2. Create venv and install
run.bat setup
run.bat install

:: 3. Activate the environment
.venv\Scripts\activate.bat

:: 4. Copy and edit environment variables
copy .env.example .env
notepad .env
```

> **Note:** `run.bat` is a CMD helper. If you prefer PowerShell, activate with `.venv\Scripts\Activate.ps1` instead (you may need `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).

---

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

python scripts/01_extract_frames.py --input_video data/raw/lec01.mp4 --video_id lec01

python scripts/02_run_asr.py --input_video data/raw/lec01.mp4 --video_id lec01

python scripts/04_build_embeddings.py --frames_dir data/interim/lec01/frames --transcript_jsonl data/interim/lec01/transcript.jsonl --video_id lec01

python scripts/05_segment_video.py --embeddings_npz data/interim/lec01/embeddings.npz --video_id lec01

python scripts/06_build_index.py --segments_jsonl data/interim/lec01/segments_pred.jsonl --video_id lec01

python scripts/07_retrieve.py --index_dir data/interim/lec01/index --query "What is ENCAPSULATION ?"
```

Or use the single-command pipeline:

```bash
# Via module
python -m vlm_video.pipelines.run_all data/raw/lec01.mp4 --exp_name lec01

# Via installed entry point (after pip install -e .)
vlm-pipeline data/raw/lec01.mp4 --exp_name lec01
```

---

## Retrieval features

### Query expansion

The retrieval script automatically expands common OOP terms with their Vietnamese equivalents before encoding. For example, querying `"What is encapsulation?"` internally becomes `"What is encapsulation? encapsulation đóng gói"` to improve recall on Vietnamese lecture content.

### Hybrid scoring

Results are re-ranked using a weighted combination of semantic similarity and keyword overlap:

- **Semantic score** (60%) — cosine similarity from the embedding index
- **Keyword score** (40%) — lexical overlap between query tokens and segment transcript

### LLM reranking with Claude

For higher-precision retrieval, pass `--rerank` to `07_retrieve.py` to call the Anthropic Claude API as a second-stage reranker:

```bash
python scripts/07_retrieve.py \
  --index_dir data/interim/lec01/index \
  --query "What is polymorphism?" \
  --rerank
```

This requires `ANTHROPIC_API_KEY` to be set in your environment or in `.env`. LLM reranking can also be enabled by default in `configs/default.yaml`:

```yaml
retrieval:
  use_llm_rerank: true
```

---

## Directory structure

```
vlm-video-segmentation/
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
│   ├── retrieval/       ← SklearnIndex, FaissIndex, factory, ranking,
│   │                       QueryEncoder, LLMReranker
│   ├── evaluation/      ← Boundary F1, retrieval metrics
│   └── pipelines/       ← EndToEndPipeline
├── tests/
│   ├── test_boundary_f1.py
│   └── test_retrieval_backend.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── run.bat              ← Windows CMD helper
└── README.md
```

---

## Configuration guide

All settings live in `configs/default.yaml`. Override any value by creating a custom YAML and passing `--config your_config.yaml` to any script.

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `frame_extraction` | `fps` | `1.0` | Frames per second to extract |
| `asr` | `model` | `large-v3` | Whisper model size |
| `asr` | `language` | `vi` | ISO language code |
| `embeddings` | `model` | `ViT-L-14` | CLIP architecture |
| `embeddings` | `pretrained` | `laion2b_s32b_b82k` | CLIP pretrained weights |
| `embeddings` | `weights.visual` | `0.6` | Visual modality weight |
| `embeddings` | `weights.text` | `0.3` | ASR text modality weight |
| `embeddings` | `weights.ocr` | `0.1` | OCR text modality weight |
| `segmentation` | `method` | `clip_latefusion` | Segmentation algorithm |
| `segmentation` | `segmentation_method` | `threshold` | Boundary detection (`threshold` or `pelt`) |
| `segmentation` | `min_segment_duration` | `30` | Merge segments shorter than this (seconds) |
| `segmentation` | `merge_sim_threshold` | `0.9` | Cosine similarity threshold for merging adjacent segments |
| `retrieval` | `backend` | `sklearn` | `sklearn` or `faiss` |
| `retrieval` | `top_k` | `5` | Results to return |
| `retrieval` | `use_llm_rerank` | `false` | Enable Claude LLM reranking by default |

See `configs/default.yaml` for all options with inline documentation.

---

## Environment variables

Copy `.env.example` to `.env` and fill in as needed:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Only for `--rerank` | Anthropic API key for Claude LLM reranking |

---

## Running tests

```bash
pytest
```

Or via the helper on Windows:

```cmd
run.bat test
```

---

## Linting

```bash
ruff check src/ scripts/ tests/
```

Or via the helper on Windows:

```cmd
run.bat lint
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{nckh2026vlm,
  title  = {Optimizing Segmentation and Event Retrieval in Syllabus Videos using VLMs},
  author = {NCKH Research Team},
  year   = {2026},
  url    = {https://github.com/anhhoangdn/vlm-video-segmentation}
}
```

---

## License

MIT License — see `LICENSE` file for details.
