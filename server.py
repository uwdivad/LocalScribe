#!/usr/bin/env python3
"""
MCP server that watches input/ for media files and transcribes them automatically.

Add to Claude Desktop's config (claude_desktop_config.json):
    {
        "mcpServers": {
            "whisper": {
                "command": "python",
                "args": ["C:/path/to/whisper/server.py"],
                "cwd": "C:/path/to/whisper"
            }
        }
    }

Environment variables (set in .env or system env):
    HF_TOKEN            — required: Hugging Face access token
    INPUT_DIR           — default: input
    OUTPUT_DIR          — default: output
    DEVICE              — default: cuda
    WHISPER_MODEL       — default: large-v3
    COMPUTE_TYPE        — default: float32
    BATCH_SIZE          — default: 16
    BEAM_SIZE           — default: 10
    LANGUAGE            — default: en
    NO_ALIGN            — default: false  (set to 1 to skip alignment)
    DIARIZATION_MODEL   — default: pyannote/speaker-diarization-community-1
"""

import os
import sys
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from transcribe import SUPPORTED_EXTENSIONS, run_transcription

# --- Config from environment ---

INPUT_DIR         = Path(os.environ.get("INPUT_DIR", "input"))
OUTPUT_DIR        = Path(os.environ.get("OUTPUT_DIR", "output"))
HF_TOKEN          = os.environ.get("HF_TOKEN")
DEVICE            = os.environ.get("DEVICE", "cuda")
MODEL             = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE      = os.environ.get("COMPUTE_TYPE", "float32")
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE", "16"))
BEAM_SIZE         = int(os.environ.get("BEAM_SIZE", "10"))
LANGUAGE          = os.environ.get("LANGUAGE", "en")
NO_ALIGN          = os.environ.get("NO_ALIGN", "").lower() in ("1", "true", "yes")
NO_DIARIZE        = os.environ.get("NO_DIARIZE", "").lower() in ("1", "true", "yes")
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Job tracking ---

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


@dataclass
class Job:
    file: Path
    status: JobStatus = JobStatus.PENDING
    queued_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: str | None = None
    finished_at: str | None = None
    output_txt: str | None = None
    output_json: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "file": self.file.name,
            "status": self.status,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_txt": self.output_txt,
            "output_json": self.output_json,
            "error": self.error,
        }


_jobs: dict[str, Job] = {}
_job_queue: queue.Queue[Job] = queue.Queue()
_lock = threading.Lock()


def _enqueue(path: Path) -> Job | None:
    """Add a file to the transcription queue. Returns None if already queued/running."""
    name = path.name
    with _lock:
        existing = _jobs.get(name)
        if existing and existing.status in (JobStatus.PENDING, JobStatus.RUNNING):
            return None
        job = Job(file=path)
        _jobs[name] = job
    _job_queue.put(job)
    return job


def _worker() -> None:
    """Background thread: pull jobs from the queue and transcribe them one at a time."""
    while True:
        job = _job_queue.get()
        with _lock:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now().isoformat(timespec="seconds")
        try:
            txt_path, json_path = run_transcription(
                input_file=job.file,
                output_dir=OUTPUT_DIR,
                model=MODEL,
                language=LANGUAGE,
                batch_size=BATCH_SIZE,
                compute_type=COMPUTE_TYPE,
                beam_size=BEAM_SIZE,
                device=DEVICE,
                diarization_model=DIARIZATION_MODEL,
                hf_token=HF_TOKEN,
                no_align=NO_ALIGN,
                no_diarize=NO_DIARIZE,
            )
            with _lock:
                job.status = JobStatus.DONE
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                job.output_txt = str(txt_path)
                job.output_json = str(json_path)
        except Exception as exc:
            with _lock:
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                job.error = str(exc)
        finally:
            _job_queue.task_done()

# --- File watcher ---

class _WatchHandler(FileSystemEventHandler):
    def _maybe_enqueue(self, path: Path) -> None:
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            print(f"[watcher] Detected: {path.name}", flush=True)
            _enqueue(path)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_enqueue(Path(event.src_path))

    def on_moved(self, event):
        # Fires when a file is moved/dropped into the watched directory
        if not event.is_directory:
            self._maybe_enqueue(Path(event.dest_path))

# --- MCP tools ---

mcp = FastMCP("Whisper Transcription Server")


@mcp.tool()
def list_input_files() -> list[str]:
    """List supported media files currently sitting in the input directory."""
    return sorted(
        p.name for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


@mcp.tool()
def queue_file(filename: str) -> str:
    """Manually queue a file from the input directory for transcription.

    Args:
        filename: Name of the file inside the input directory (e.g. 'interview.mp4').
    """
    path = INPUT_DIR / filename
    if not path.exists():
        return f"Error: '{filename}' not found in {INPUT_DIR}/"
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return f"Error: unsupported file type '{path.suffix}'."
    job = _enqueue(path)
    if job is None:
        return f"'{filename}' is already queued or running."
    return f"Queued '{filename}' for transcription."


@mcp.tool()
def list_jobs() -> list[dict]:
    """Return all transcription jobs and their current status."""
    with _lock:
        return [j.to_dict() for j in _jobs.values()]


@mcp.tool()
def read_transcript(filename: str) -> str:
    """Read the plain-text transcript for a given input file.

    Args:
        filename: The original input filename (e.g. 'interview.mp4').
    """
    stem = Path(filename).stem
    txt_path = OUTPUT_DIR / f"{stem}_transcript.txt"
    if not txt_path.exists():
        return f"No transcript found for '{filename}'. Check list_jobs() for status."
    return txt_path.read_text(encoding="utf-8")


# --- Entry point ---

if __name__ == "__main__":
    if not NO_DIARIZE and not HF_TOKEN:
        print(
            "ERROR: HF_TOKEN not set. Add it to .env or set the environment variable.\n"
            "       Set NO_DIARIZE=1 to skip speaker labeling and run fully locally.",
            file=sys.stderr,
        )
        sys.exit(1)

    threading.Thread(target=_worker, daemon=True).start()

    observer = Observer()
    observer.schedule(_WatchHandler(), str(INPUT_DIR), recursive=False)
    observer.start()
    print(f"[server] Watching {INPUT_DIR.resolve()} for new media files ...", flush=True)

    mcp.run()
