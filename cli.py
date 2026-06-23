#!/usr/bin/env python3
"""
Command-line interface for the Whisper transcription pipeline.

Usage:
    1. Create a .env file in this directory:
           HF_TOKEN=hf_...
    2. Run:
           python cli.py input/interview.mp4

    Or set HF_TOKEN in your environment directly (export / set / $env:).
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
from transcribe import SUPPORTED_EXTENSIONS, run_transcription


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video and label two speakers as Person1/Person2."
    )
    parser.add_argument("input_file", type=Path, help="Path to media file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "output")),
        help="Directory for output files (default: $OUTPUT_DIR or output/)",
    )
    parser.add_argument("--model", default="large-v3", help="Whisper model name")
    parser.add_argument(
        "--language", default="en", help="BCP-47 language code (default: en)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Inference batch size (lower = less VRAM)"
    )
    parser.add_argument(
        "--compute-type",
        default="float32",
        choices=["float32", "float16", "bfloat16", "int8"],
        help="Model compute type (default: float32 for best accuracy)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="Beam search width (higher = more accurate, slower)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="pyannote diarization model on Hugging Face",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (default: $HF_TOKEN or .env)",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip word-level alignment (faster, less precise timestamps)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization — no HF token required, fully local",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        parser.error(f"File not found: {args.input_file}")

    suffix = args.input_file.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        parser.error(
            f"Unsupported extension '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not args.no_diarize and not args.hf_token:
        parser.error(
            "Hugging Face token required for diarization. Set HF_TOKEN in .env, as an env var, or pass --hf-token.\n"
            "Get a token at: https://huggingface.co/settings/tokens\n"
            "Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "Or use --no-diarize to skip speaker labeling entirely (no token needed)."
        )

    return args


def main() -> None:
    args = parse_args()
    try:
        run_transcription(
            input_file=args.input_file,
            output_dir=args.output_dir,
            model=args.model,
            language=args.language,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            device=args.device,
            diarization_model=args.diarization_model,
            hf_token=args.hf_token,
            no_align=args.no_align,
            no_diarize=args.no_diarize,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:  # type: ignore[name-defined]
        print(
            "\nOOM: GPU out of memory. Try:\n"
            "  --batch-size 8    (reduce from default 16)\n"
            "  --compute-type int8\n"
            "  --no-align        (skip alignment step)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
