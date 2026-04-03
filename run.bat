@echo off
setlocal enabledelayedexpansion

REM run.bat — Windows CMD helper for vlm_video research project
REM Usage: run.bat [setup|install|test|lint|ffmpeg|help]

set "CMD=%~1"
if "%CMD%"=="" set "CMD=help"

set "VENV_DIR=.venv"
set "PYTHON=python"

goto :main

:header
echo.
echo ==> %~1
goto :eof

:assert_python
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found. Install from https://www.python.org/downloads/
  exit /b 1
)
for /f "delims=" %%v in ('%PYTHON% --version 2^>^&1') do echo Found: %%v
goto :eof

:ensure_venv
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [ERROR] Virtual environment "%VENV_DIR%" not found. Run: run.bat setup
  exit /b 1
)
goto :eof

:setup_venv
call :header "Creating virtual environment in %VENV_DIR%"
call :assert_python || exit /b 1

if not exist "%VENV_DIR%\Scripts\python.exe" (
  %PYTHON% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
  echo Virtual environment created.
) else (
  echo Virtual environment already exists, skipping creation.
)
goto :eof

:install_package
call :header "Installing vlm_video package (dev dependencies)"
call :ensure_venv || exit /b 1

"%VENV_DIR%\Scripts\pip.exe" install --upgrade pip
if errorlevel 1 exit /b 1

"%VENV_DIR%\Scripts\pip.exe" install -e ".[dev]"
if errorlevel 1 exit /b 1

echo.
echo Optional extras:
echo   OCR support : "%VENV_DIR%\Scripts\pip.exe" install -e ".[ocr]"
echo   FAISS index : "%VENV_DIR%\Scripts\pip.exe" install -e ".[faiss]"
echo.
echo Package installed.
goto :eof

:run_tests
call :header "Running tests"
call :ensure_venv || exit /b 1

if not exist "%VENV_DIR%\Scripts\pytest.exe" (
  echo [ERROR] pytest not found. Run: run.bat install
  exit /b 1
)
"%VENV_DIR%\Scripts\pytest.exe" tests/ -v
exit /b %errorlevel%

:run_lint
call :header "Running ruff linter"
call :ensure_venv || exit /b 1

if not exist "%VENV_DIR%\Scripts\ruff.exe" (
  echo [ERROR] ruff not found. Run: run.bat install
  exit /b 1
)
"%VENV_DIR%\Scripts\ruff.exe" check src/ scripts/ tests/
exit /b %errorlevel%

:ffmpeg_hint
call :header "FFmpeg installation hint (Windows)"
echo   winget : winget install ffmpeg
echo   choco  : choco install ffmpeg
echo   manual : https://www.gyan.dev/ffmpeg/builds/
echo   After installing, make sure ffmpeg.exe is in your PATH.
goto :eof

:show_help
echo.
echo vlm_video research project — CMD helper
echo Usage: run.bat [command]
echo.
echo Commands:
echo   setup    Create Python virtual environment
echo   install  Install package and dev dependencies into venv
echo   test     Run pytest test suite
echo   lint     Run ruff linter
echo   ffmpeg   Show Windows ffmpeg install instructions
echo   help     Show this help message (default)
echo.
echo Quick start:
echo   run.bat setup
echo   run.bat install
echo   .venv\Scripts\activate.bat
echo.
echo End-to-end pipeline (single video):
echo   python scripts\01_extract_frames.py --input_video data\raw\lecture.mp4 --video_id lec01
echo   python scripts\02_run_asr.py         --input_video data\raw\lecture.mp4 --video_id lec01
echo   python scripts\03_run_ocr.py         --frames_dir data\interim\lec01\frames --video_id lec01
echo   python scripts\04_build_embeddings.py --frames_dir data\interim\lec01\frames --video_id lec01 --transcript_jsonl data\interim\lec01\transcript.jsonl
echo   python scripts\05_segment_video.py    --embeddings_npz data\interim\lec01\embeddings.npz --video_id lec01
echo   python scripts\06_build_index.py      --segments_jsonl data\interim\lec01\segments_pred.jsonl --video_id lec01
echo   python scripts\07_retrieve.py         --index_dir data\interim\lec01\index --query "What is gradient descent?"
goto :eof

:main
if /i "%CMD%"=="setup"   (call :setup_venv & exit /b %errorlevel%)
if /i "%CMD%"=="install" (call :setup_venv & if errorlevel 1 exit /b 1 & call :install_package & exit /b %errorlevel%)
if /i "%CMD%"=="test"    (call :run_tests)
if /i "%CMD%"=="lint"    (call :run_lint)
if /i "%CMD%"=="ffmpeg"  (call :ffmpeg_hint & exit /b %errorlevel%)
if /i "%CMD%"=="help"    (call :show_help & exit /b 0)

echo [ERROR] Unknown command: %CMD%
call :show_help
exit /b 1
