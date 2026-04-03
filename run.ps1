# run.ps1 — Windows PowerShell helper for vlm_video research project
# Usage:  .\run.ps1 [setup|install|test|lint|help]
# Requires: Python 3.11+, winget or choco (optional for ffmpeg)

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$VENV_DIR = ".venv"
$PYTHON   = "python"

function Write-Header($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Assert-Python {
    try {
        $ver = & $PYTHON --version 2>&1
        Write-Host "Found: $ver"
    } catch {
        Write-Error "Python not found. Install from https://www.python.org/downloads/"
        exit 1
    }
}

function Setup-Venv {
    Write-Header "Creating virtual environment in $VENV_DIR"
    Assert-Python
    if (-not (Test-Path $VENV_DIR)) {
        & $PYTHON -m venv $VENV_DIR
        Write-Host "Virtual environment created." -ForegroundColor Green
    } else {
        Write-Host "Virtual environment already exists, skipping creation."
    }
}

function Install-Package {
    Write-Header "Installing vlm_video package (CPU extras)"
    $pip = "$VENV_DIR\Scripts\pip.exe"
    & $pip install --upgrade pip
    & $pip install -e ".[dev]"
    Write-Host ""
    Write-Host "Optional extras:" -ForegroundColor Yellow
    Write-Host "  OCR support : $pip install -e '.[ocr]'"
    Write-Host "  FAISS index : $pip install -e '.[faiss]'"
    Write-Host ""
    Write-Host "Package installed." -ForegroundColor Green
}

function Show-FfmpegHint {
    Write-Header "FFmpeg installation hint (Windows)"
    Write-Host "  winget : winget install ffmpeg" -ForegroundColor Yellow
    Write-Host "  choco  : choco install ffmpeg" -ForegroundColor Yellow
    Write-Host "  manual : https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Yellow
    Write-Host "  After installing, make sure ffmpeg.exe is in your PATH."
}

function Run-Tests {
    Write-Header "Running tests"
    $pytest = "$VENV_DIR\Scripts\pytest.exe"
    & $pytest tests/ -v
}

function Run-Lint {
    Write-Header "Running ruff linter"
    $ruff = "$VENV_DIR\Scripts\ruff.exe"
    & $ruff check src/ scripts/ tests/
}

function Show-Help {
    Write-Host ""
    Write-Host "vlm_video research project — PowerShell helper" -ForegroundColor Cyan
    Write-Host "Usage: .\run.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  setup    Create Python virtual environment"
    Write-Host "  install  Install package and dev dependencies into venv"
    Write-Host "  test     Run pytest test suite"
    Write-Host "  lint     Run ruff linter"
    Write-Host "  ffmpeg   Show Windows ffmpeg install instructions"
    Write-Host "  help     Show this help message (default)"
    Write-Host ""
    Write-Host "Quick start:" -ForegroundColor Green
    Write-Host "  .\run.ps1 setup"
    Write-Host "  .\run.ps1 install"
    Write-Host "  .venv\Scripts\Activate.ps1"
    Write-Host ""
    Write-Host "End-to-end pipeline (single video):" -ForegroundColor Green
    Write-Host "  python scripts\01_extract_frames.py --input_video data\raw\lecture.mp4 --video_id lec01"
    Write-Host "  python scripts\02_run_asr.py         --input_video data\raw\lecture.mp4 --video_id lec01"
    Write-Host "  python scripts\03_run_ocr.py          --frames_dir data\interim\lec01\frames --video_id lec01"
    Write-Host "  python scripts\04_build_embeddings.py --frames_dir data\interim\lec01\frames --video_id lec01"
    Write-Host "                                         --transcript_jsonl data\interim\lec01\transcript.jsonl"
    Write-Host "  python scripts\05_segment_video.py    --embeddings_npz data\interim\lec01\embeddings.npz --video_id lec01"
    Write-Host "  python scripts\06_build_index.py      --segments_jsonl data\interim\lec01\segments_pred.jsonl --video_id lec01"
    Write-Host "  python scripts\07_retrieve.py         --index_dir data\interim\lec01\index --query `"What is gradient descent?`""
    Write-Host ""
}

switch ($Command.ToLower()) {
    "setup"   { Setup-Venv }
    "install" { Setup-Venv; Install-Package }
    "test"    { Run-Tests }
    "lint"    { Run-Lint }
    "ffmpeg"  { Show-FfmpegHint }
    default   { Show-Help }
}
