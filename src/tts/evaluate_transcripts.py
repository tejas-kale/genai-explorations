"""CLI tool: transcribe an MP3 and evaluate it against a reference transcript."""

import argparse
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model
from mlx_lm import generate, load
from pydub import AudioSegment

ASR_MODEL_DEFAULT = "mlx-community/Qwen3-ASR-1.7B-8bit"
LLM_MODEL_DEFAULT = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# 5-minute chunks keep audio features well under Metal's 4 GB buffer limit.
CHUNK_MS = 5 * 60 * 1000

EVAL_PROMPT_TEMPLATE = """\
You are evaluating an ASR-generated transcript against a reference transcript of the same audio.

## Reference transcript
{reference}

## Generated transcript
{generated}

Evaluate the generated transcript on the following three dimensions. For each, give a \
rating (Excellent / Good / Fair / Poor) and a brief explanation with specific examples \
where relevant.

### 1. Grammatical correctness
Are sentences well-formed? Note any errors, garbled words, or run-ons.

### 2. Completeness
Is all speech from the reference captured? Note any missing sentences, segments, or speakers.

### 3. Preservation of main points, arguments, and conclusion
Are the core ideas, causal arguments, and conclusion retained? Note any omissions or \
distortions of substance.

### Summary
One paragraph overall assessment.
"""


def _transcribe(audio_path: Path, asr_model_id: str) -> str:
    """Transcribe audio file using the given ASR model.

    Long files are split into CHUNK_MS-sized pieces to stay within Metal's
    per-buffer memory limit before transcribing each chunk in turn.
    """
    print(f"Loading ASR model: {asr_model_id}", file=sys.stderr)
    model = load_model(asr_model_id)

    audio = AudioSegment.from_file(audio_path)
    duration_s = len(audio) / 1000
    chunks = [audio[start : start + CHUNK_MS] for start in range(0, len(audio), CHUNK_MS)]
    n = len(chunks)
    print(
        f"Transcribing {duration_s:.0f}s of audio in {n} chunk(s)…",
        file=sys.stderr,
    )

    parts: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, chunk in enumerate(chunks, 1):
            print(f"  chunk {i}/{n}", file=sys.stderr)
            chunk_path = Path(tmp) / f"chunk_{i:04d}.mp3"
            chunk.export(chunk_path, format="mp3")
            result = generate_transcription(model=model, audio=str(chunk_path))
            parts.append(result.text)

    del model
    mx.metal.clear_cache()
    return " ".join(parts)


def _evaluate(reference: str, generated: str, llm_model_id: str) -> str:
    """Compare transcripts using an LLM and return the evaluation text."""
    print(f"Loading LLM model: {llm_model_id}", file=sys.stderr)
    llm, tokenizer = load(llm_model_id)
    prompt_text = EVAL_PROMPT_TEMPLATE.format(reference=reference, generated=generated)
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("Evaluating…", file=sys.stderr)
    comparison = generate(llm, tokenizer, prompt=prompt, max_tokens=1500, verbose=False)
    del llm, tokenizer
    mx.metal.clear_cache()
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an MP3 file and evaluate the result against a reference "
            "transcript, printing the comparison to stdout."
        )
    )
    parser.add_argument("audio", type=Path, help="Path to the MP3 file.")
    parser.add_argument(
        "reference",
        type=Path,
        help="Path to the reference transcript (plain text or Markdown).",
    )
    parser.add_argument(
        "--asr-model",
        default=ASR_MODEL_DEFAULT,
        metavar="MODEL_ID",
        help=f"Hugging Face model ID for the ASR model (default: {ASR_MODEL_DEFAULT}).",
    )
    parser.add_argument(
        "--llm-model",
        default=LLM_MODEL_DEFAULT,
        metavar="MODEL_ID",
        help=f"Hugging Face model ID for the evaluation LLM (default: {LLM_MODEL_DEFAULT}).",
    )

    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"Audio file not found: {args.audio}")
    if not args.reference.exists():
        parser.error(f"Reference transcript not found: {args.reference}")

    generated = _transcribe(args.audio, args.asr_model)
    reference = args.reference.read_text()
    comparison = _evaluate(reference, generated, args.llm_model)

    print(comparison)


if __name__ == "__main__":
    main()
