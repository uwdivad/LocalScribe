# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-07-21

Initial release.

### Added
- Core transcription pipeline (`transcribe.py`): ffmpeg audio extraction,
  WhisperX transcription, word-level alignment, and optional two-speaker
  diarization via pyannote with `Person1`/`Person2` labeling.
- CLI (`cli.py`) wrapping `run_transcription()` with `--no-align` and
  `--no-diarize` options for fully local, token-free runs.
- MCP server (`server.py`) that watches `input/` and auto-transcribes dropped
  files, exposing `list_input_files`, `queue_file`, `list_jobs`, and
  `read_transcript` tools.
- GitHub Actions CI (ruff lint + byte-compile across Python 3.11–3.13) and a
  tag-triggered release workflow.
