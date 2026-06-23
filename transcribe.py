"""
Core transcription pipeline. Import and call run_transcription() directly,
or use cli.py for command-line use, or server.py for the MCP watcher server.
"""

import gc
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def extract_audio(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-f", "wav",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it and make sure it is on your PATH.\n"
            "  Linux:   sudo apt install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg  (or https://ffmpeg.org/download.html)"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting audio:\n{result.stderr.decode(errors='replace')}"
        )


def load_whisper_model(model_name: str, device: str, compute_type: str, beam_size: int):
    import whisperx

    print(f"Loading Whisper model '{model_name}' on {device} ...")
    t0 = time.time()
    model = whisperx.load_model(
        model_name, device=device, compute_type=compute_type,
        asr_options={"beam_size": beam_size},
    )
    print(f"  Whisper loaded in {time.time() - t0:.1f}s")
    return model


def load_diarization_model(diarization_model_name: str, device: str, hf_token: str):
    print(f"Loading diarization model '{diarization_model_name}' ...")
    t0 = time.time()
    from pyannote.audio import Pipeline
    model = Pipeline.from_pretrained(diarization_model_name, token=hf_token)
    model = model.to(torch.device(device))
    print(f"  Diarization loaded in {time.time() - t0:.1f}s")
    return model


def transcribe_audio(model, audio, batch_size: int, language: str | None) -> dict:
    import whisperx

    print("Transcribing ...")
    t0 = time.time()
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected = result.get("language", "unknown")
    print(f"  Transcription done in {time.time() - t0:.1f}s (language: {detected})")
    return result


def align_transcription(result: dict, audio, device: str, no_align: bool) -> dict:
    import whisperx

    if no_align:
        return result

    language = result.get("language", "en")
    print(f"Aligning word timestamps (language={language}) ...")
    t0 = time.time()
    try:
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False
        )
        del model_a
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Alignment done in {time.time() - t0:.1f}s")
    except Exception as exc:
        print(f"  WARNING: Alignment failed ({exc}); continuing with unaligned timestamps.")

    return result


def run_diarization(diarize_model, audio_path: Path) -> Any | None:
    print("Running speaker diarization ...")
    t0 = time.time()
    try:
        import pandas as pd
        diarization = diarize_model(str(audio_path), min_speakers=2, max_speakers=2)
        print(f"  Diarization done in {time.time() - t0:.1f}s")

        annotation = getattr(diarization, "speaker_diarization", diarization)
        diarize_df = pd.DataFrame(
            annotation.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
        return diarize_df
    except Exception as exc:
        print(f"  WARNING: Diarization failed ({exc}); output will have no speaker labels.")
        return None


def assign_speakers(diarize_segments, result: dict) -> tuple[list[dict], dict[str, str]]:
    import whisperx

    segments = result.get("segments", [])

    if diarize_segments is None:
        for seg in segments:
            seg["speaker"] = "Unknown"
        return segments, {}

    result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)
    segments = result_with_speakers.get("segments", segments)

    all_speakers: list[str] = []
    for seg in segments:
        sp = seg.get("speaker")
        if sp:
            all_speakers.append(sp)
        for word in seg.get("words", []):
            sp = word.get("speaker")
            if sp:
                all_speakers.append(sp)

    unique_speakers = sorted(set(all_speakers))

    if not unique_speakers:
        print("  WARNING: No speaker labels assigned; labeling all segments as [Unknown].")
        for seg in segments:
            seg["speaker"] = "Unknown"
            for word in seg.get("words", []):
                word["speaker"] = "Unknown"
        return segments, {}

    if len(unique_speakers) > 2:
        from collections import Counter
        counts = Counter(all_speakers)
        top2 = [sp for sp, _ in counts.most_common(2)]
        extras = [sp for sp in unique_speakers if sp not in top2]
        print(f"  WARNING: {len(unique_speakers)} speakers detected; merging {extras} into Person2.")
        mapping = {top2[0]: "Person1", top2[1]: "Person2"}
        for extra in extras:
            mapping[extra] = "Person2"
    else:
        mapping = (
            {unique_speakers[0]: "Person1", unique_speakers[1]: "Person2"}
            if len(unique_speakers) == 2
            else {unique_speakers[0]: "Person1"}
        )

    print(f"  Speaker mapping: {mapping}")

    for seg in segments:
        raw = seg.get("speaker", "")
        seg["speaker"] = mapping.get(raw, raw or "Unknown")
        for word in seg.get("words", []):
            raw_w = word.get("speaker", "")
            word["speaker"] = mapping.get(raw_w, raw_w or "Unknown")

    return segments, mapping


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        start = _fmt_time(seg.get("start", 0.0))
        end = _fmt_time(seg.get("end", 0.0))
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        lines.append(f"[{start} --> {end}]  {speaker}: {text}")
    return "\n".join(lines)


def write_output(
    segments: list[dict],
    speaker_mapping: dict[str, str],
    input_path: Path,
    output_dir: Path,
    transcript_text: str,
) -> tuple[Path, Path]:
    stem = input_path.stem
    txt_path = output_dir / f"{stem}_transcript.txt"
    json_path = output_dir / f"{stem}_transcript.json"

    txt_path.write_text(transcript_text, encoding="utf-8")

    payload = {
        "input_file": str(input_path),
        "speaker_mapping": speaker_mapping,
        "segments": segments,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return txt_path, json_path


def run_transcription(
    input_file: Path,
    output_dir: Path = Path("output"),
    *,
    model: str = "large-v3",
    language: str = "en",
    batch_size: int = 16,
    compute_type: str = "float32",
    beam_size: int = 10,
    device: str = "cuda",
    diarization_model: str = "pyannote/speaker-diarization-community-1",
    hf_token: str | None = None,
    no_align: bool = False,
    no_diarize: bool = False,
) -> tuple[Path, Path]:
    """Run the full transcription pipeline. Returns (txt_path, json_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.time()
    suffix = input_file.suffix.lower()
    is_video = suffix in VIDEO_EXTENSIONS

    tmp_wav_fd, tmp_wav_path_str = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_wav_fd)
    tmp_wav_path = Path(tmp_wav_path_str)

    try:
        if is_video:
            print(f"Extracting audio from video: {input_file.name}")
            extract_audio(input_file, tmp_wav_path)
            audio_path = tmp_wav_path
        else:
            audio_path = input_file
            tmp_wav_path.unlink(missing_ok=True)

        import whisperx

        print(f"Loading audio: {audio_path}")
        audio = whisperx.load_audio(str(audio_path))

        whisper_model = load_whisper_model(model, device, compute_type, beam_size)
        result = transcribe_audio(whisper_model, audio, batch_size, language)
        del whisper_model
        gc.collect()
        torch.cuda.empty_cache()

        result = align_transcription(result, audio, device, no_align)

        if no_diarize:
            diarize_segments = None
        else:
            diarize_model = load_diarization_model(diarization_model, device, hf_token)
            diarize_path = audio_path if not is_video else tmp_wav_path
            diarize_segments = run_diarization(diarize_model, diarize_path)
            del diarize_model
            gc.collect()
            torch.cuda.empty_cache()

        segments, speaker_mapping = assign_speakers(diarize_segments, result)

        transcript_text = format_transcript(segments)
        print("\n" + "=" * 72)
        print(transcript_text)
        print("=" * 72 + "\n")

        txt_path, json_path = write_output(
            segments, speaker_mapping, input_file, output_dir, transcript_text
        )

        print(f"Transcript written to: {txt_path}")
        print(f"JSON written to:       {json_path}")
        print(f"Total time: {time.time() - wall_start:.1f}s")

        return txt_path, json_path

    finally:
        tmp_wav_path.unlink(missing_ok=True)
