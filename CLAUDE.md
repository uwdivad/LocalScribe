# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LocalScribe is a local GPU-accelerated audio/video transcription tool with optional two-speaker diarization. Built on WhisperX + pyannote. Runs as either a CLI tool (`cli.py`) or an MCP server (`server.py`) that watches a directory and auto-transcribes files dropped into it.

## Setup

Install PyTorch first (not in requirements.txt):
```
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install everything else:
```
pip install -r requirements.txt
```

ffmpeg must be on PATH (`sudo apt install ffmpeg` / `winget install Gyan.FFmpeg`).

Create a `.env` file:
```
HF_TOKEN=hf_...
```

## Running

CLI:
```
python cli.py input/interview.mp4
python cli.py input/interview.mp4 --output-dir /some/other/dir --no-align
```

MCP server (watches `input/` and transcribes automatically):
```
python server.py
```

## Architecture

**`transcribe.py`** — pure library, no I/O side-effects at import. The single public entry point is:
```python
run_transcription(input_file, output_dir, *, model, language, batch_size,
                  compute_type, beam_size, device, diarization_model, hf_token, no_align)
    -> tuple[Path, Path]  # (txt_path, json_path)
```
Internally: extract audio with ffmpeg → load WhisperX + pyannote models → transcribe → align word timestamps → diarize → assign Person1/Person2 labels → write `{stem}_transcript.txt` and `{stem}_transcript.json` to `output_dir`.

**`cli.py`** — argparse wrapper around `run_transcription()`. Defaults for `--output-dir` come from the `OUTPUT_DIR` env var (falls back to `output/`). Loads `.env` before parsing args so env vars are populated.

**`server.py`** — FastMCP server. On startup it launches a background worker thread (single worker, so GPU jobs are serialised) and a watchdog observer on `INPUT_DIR`. Dropping a supported file into `input/` auto-enqueues it. Exposes four MCP tools: `list_input_files`, `queue_file`, `list_jobs`, `read_transcript`. Job state lives in the in-process `_jobs` dict (not persisted across restarts).

## Key constraints

- `huggingface-hub` is pinned to `0.36.2` (latest 0.x) because whisperX 3.8.5 requires `<1.0.0`.
- `triton` is a Linux/macOS-only PyTorch dependency — it is absent on Windows and should not be added to requirements.txt.
- The worker in `server.py` processes one job at a time intentionally — the GPU can't handle concurrent whisperX + pyannote inference.
- Diarization is optional (`--no-diarize` / `NO_DIARIZE=1`). Without it, no HF token is needed and the pipeline is fully local after the initial Whisper model download. `load_whisper_model()` and `load_diarization_model()` are separate functions in `transcribe.py` for this reason.

## Configurable via `.env`

`server.py` reads all config from environment variables. `cli.py` reads `OUTPUT_DIR`. Relevant keys:

| Key | Default |
|-----|---------|
| `HF_TOKEN` | — (required for diarization) |
| `INPUT_DIR` | `input` |
| `OUTPUT_DIR` | `output` |
| `DEVICE` | `cuda` |
| `WHISPER_MODEL` | `large-v3` |
| `COMPUTE_TYPE` | `float32` |
| `BATCH_SIZE` | `16` |
| `BEAM_SIZE` | `10` |
| `LANGUAGE` | `en` |
| `NO_ALIGN` | `false` |
| `NO_DIARIZE` | `false` |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` |
