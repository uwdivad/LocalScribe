# Repository Guidelines

## Project Structure & Module Organization

This repository is a small Python transcription service built around WhisperX and pyannote. The main entry points are `cli.py` for one-off command-line transcription and `server.py` for the MCP watcher server. Shared transcription logic lives in `transcribe.py`; keep pipeline behavior there rather than duplicating it in entry points. Place source media in `input/` and generated transcript artifacts in `output/`. Do not commit local secrets from `.env`, virtual environments, IDE metadata, or large generated media/transcripts unless intentionally needed as fixtures.

## Build, Test, and Development Commands

Create and activate a virtual environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install CUDA-enabled PyTorch first, then project dependencies:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Run a local transcription:

```bash
python cli.py input/example.mp4 --no-diarize
```

Start the MCP watcher server:

```bash
python server.py
```

Install `ffmpeg` separately and ensure it is on `PATH`; video processing depends on it.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints for public helpers, and `pathlib.Path` for filesystem paths. Keep environment-derived configuration centralized near the top of `server.py`. Prefer small functions in `transcribe.py` that expose one pipeline step, using snake_case for functions and variables and UPPER_CASE for constants such as `SUPPORTED_EXTENSIONS`.

## Testing Guidelines

There is no automated test suite in the current tree. For changes, run the smallest practical smoke test with a short media file, preferably `--no-diarize` when the change does not involve speaker labels. Verify both expected outputs are written: `output/<stem>_transcript.txt` and `output/<stem>_transcript.json`. When adding tests, use `tests/` with `test_*.py` names and mock WhisperX, pyannote, CUDA, and ffmpeg calls instead of requiring large models.

## Commit & Pull Request Guidelines

No project-specific Git history convention is available from this checkout. Use concise imperative commit subjects, for example `Add transcript smoke test`. Pull requests should describe the behavioral change, list manual or automated checks run, mention GPU/CPU and diarization coverage, and link related issues. Include screenshots only for UI-facing MCP client changes.

## Security & Configuration Tips

Keep `HF_TOKEN` in `.env` or the process environment only. Do not log tokens, commit `.env`, or include private audio/video in examples.
