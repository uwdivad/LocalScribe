# LocalScribe

Local GPU-accelerated audio/video transcription with timestamped transcripts and optional speaker diarization.

Built on [WhisperX](https://github.com/m-bain/whisperX) + [pyannote.audio](https://github.com/pyannote/pyannote-audio), runs on your local GPU.

---

## ✨ Features

- 🎬 Supports MP4, MKV, AVI, MP3, WAV, M4A, FLAC
- 👥 Optional two-speaker diarization — labels output as **Person1** / **Person2**
- ⚡ Word-level timestamp alignment
- 🖥️ Works on Linux (WSL) and Windows
- 🤖 MCP server mode — watch a folder and transcribe files automatically
- 📄 Outputs both `.txt` (human-readable) and `.json` (structured, with word timestamps)

---

## 🚀 Setup

### 1. Install PyTorch with CUDA 12.8

```bash
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg

```bash
# Linux / WSL
sudo apt install ffmpeg

# Windows
winget install Gyan.FFmpeg
```

### 4. Configure credentials *(required for diarization only)*

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_token_here
```

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the model terms at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

> **Running without diarization?** No token or internet connection needed at runtime. Use `--no-diarize` (CLI) or `NO_DIARIZE=1` (server). Whisper models are downloaded once on first use and cached locally.

---

## 🎯 Usage

### CLI

```bash
python cli.py input/interview.mp4
```

```
Options:
  --output-dir PATH       Where to write transcripts (default: output/)
  --model NAME            Whisper model (default: large-v3)
  --language CODE         BCP-47 language code (default: en)
  --batch-size N          Inference batch size, lower = less VRAM (default: 16)
  --compute-type TYPE     float32 / float16 / bfloat16 / int8 (default: float32)
  --beam-size N           Beam search width (default: 10)
  --device cuda|cpu       (default: cuda)
  --no-align              Skip word-level alignment, faster but less precise
  --no-diarize            Skip speaker diarization — no HF token needed, fully local
  --hf-token TOKEN        Override HF_TOKEN from .env
```

### 🤖 MCP Server

Start the server and it will watch `input/` for new files, transcribing them automatically:

```bash
python server.py
```

**MCP tools exposed:**

| Tool | Description |
|------|-------------|
| `list_input_files` | List media files waiting in `input/` |
| `queue_file` | Manually queue a specific file |
| `list_jobs` | See all jobs and their status |
| `read_transcript` | Read a completed transcript |

**Wire it into Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "whisper": {
      "command": "python",
      "args": ["C:/path/to/whisper/server.py"],
      "cwd": "C:/path/to/whisper"
    }
  }
}
```

---

## ⚙️ Configuration

All settings can be set in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required for diarization)* | Hugging Face access token |
| `INPUT_DIR` | `input` | Directory to watch *(server only)* |
| `OUTPUT_DIR` | `output` | Directory for transcripts *(server + CLI)* |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `COMPUTE_TYPE` | `float32` | float32 / float16 / bfloat16 / int8 |
| `BATCH_SIZE` | `16` | Lower if you hit VRAM limits |
| `BEAM_SIZE` | `10` | Higher = more accurate, slower |
| `LANGUAGE` | `en` | BCP-47 language code |
| `NO_ALIGN` | `false` | Set to `1` to skip word alignment |
| `NO_DIARIZE` | `false` | Set to `1` to skip diarization (no token needed) |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | pyannote model |

---

## 📁 Output

For each input file, two files are written to `output/`:

**`interview_transcript.txt`** — human-readable:
```
[00:00:01.280 --> 00:00:04.560]  Person1: Hey, can you hear me okay?
[00:00:04.820 --> 00:00:06.140]  Person2: Yeah, loud and clear.
```

**`interview_transcript.json`** — structured with word-level timestamps and speaker mapping.

---

## 🛠️ Troubleshooting

**Out of GPU memory:**
```bash
python cli.py input/file.mp4 --batch-size 8 --compute-type int8 --no-align
```

**ffmpeg not found:** Make sure `ffmpeg` is installed and on your `PATH`.

**Token errors:** Ensure your `HF_TOKEN` is valid and you've accepted the pyannote model terms on Hugging Face. Or skip diarization entirely with `--no-diarize` — no token needed.
