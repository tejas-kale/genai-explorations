"""CLI tool: transcribe, diarise, attribute speakers, and evaluate an MP3.

Pipeline
--------
1. Split audio into overlapping chunks (5 min stride, 30 s overlap).
2. Per chunk: Whisper ASR (segment timestamps) + SortFormer diarisation →
   max-overlap speaker assignment → grouped turns with absolute timestamps.
3. At each chunk boundary the LLM reconciles the duplicated ~30 s overlap:
   it deduplicates content and returns a speaker-label mapping so labels are
   kept consistent across the whole episode.
4. The final speaker-attributed transcript is evaluated against a reference
   on four dimensions using a small instruction-tuned LLM.

Why Whisper instead of Qwen3-ASR?
    Qwen3-ASR timestamps require Qwen3-ForcedAligner-0.6B, which has no MLX
    port and only runs on CUDA. Whisper natively returns per-segment
    (start, end, text) timestamps, enabling direct time-overlap alignment.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model as load_asr_model
from mlx_audio.vad import load as load_diar_model
from mlx_lm import generate, load
from pydub import AudioSegment
from transformers import WhisperProcessor

ASR_MODEL_DEFAULT = "mlx-community/whisper-large-v3-turbo"
DIAR_MODEL_DEFAULT = "mlx-community/diar_sortformer_4spk-v1-fp32"
LLM_MODEL_DEFAULT = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# Non-overlapping stride and the trailing overlap added to each chunk so that
# adjacent chunks share ~30 s of audio for boundary reconciliation.
STRIDE_MS = 5 * 60 * 1000
OVERLAP_MS = 30 * 1000

# How many turns to extract from each side of a boundary for the LLM.
BOUNDARY_TURNS = 8

# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #

_RECONCILE_PROMPT = """\
You are reconciling two overlapping sections of a speaker-attributed transcript. \
Both sections were produced from the SAME ~30 seconds of audio by independent \
transcription passes, so they contain the same speech with possibly different \
speaker labels and minor wording differences.

## End of previous section
{tail}

## Start of next section (same audio, different speaker labels)
{head}

Your tasks:
1. Produce a single clean reconciled version of this ~30-second passage. \
   Use the speaker labels from the **previous section**. \
   Fix any obvious transcription errors visible through comparison.
2. List the speaker-label mapping: which label in the next section corresponds \
   to which label in the previous section. Include only labels that appear above.

Respond in exactly this format — no extra prose:

## RECONCILED TURNS
**<Speaker X>**: <text>
(one line per turn)

## SPEAKER MAPPING
<Next label> → <Previous label>
(one line per mapping; if a next-section speaker has no match write: <Next label> → Unknown)
"""

_EVAL_PROMPT = """\
You are evaluating a speaker-attributed transcript produced by aligning ASR \
output with speaker diarisation. The goal is to assess whether the transcribed \
text is correctly attributed to the right speakers.

## Reference transcript
{reference}

## Speaker-attributed transcript
{attributed}

Evaluate the speaker-attributed transcript on the following four dimensions. \
For each, give a rating (Excellent / Good / Fair / Poor) and a brief explanation \
with specific examples where relevant.

### 1. Speaker count
Does the number of distinct speakers match the reference? Note any over- or \
under-clustering.

### 2. Speaker consistency
Is each speaker label used consistently throughout? Note any turns where the \
same speaker from the reference appears under different labels.

### 3. Turn-taking accuracy
Do the speaker turn boundaries align with the speaker changes implied by the \
reference? Note any missed transitions or spurious splits.

### 4. Text–speaker alignment
Is the transcribed text attributed to the correct speaker? Note any segments \
where the content is clearly from the wrong speaker.

### Summary
One paragraph overall assessment of the speaker attribution quality.
"""


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

Turn = dict  # keys: speaker (str), text (str), start_s (float), end_s (float)


# --------------------------------------------------------------------------- #
# ASR + diarisation helpers
# --------------------------------------------------------------------------- #

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _assign_speakers(asr_segments: list, diar_segments: list, offset_s: float) -> list[Turn]:
    """Assign a speaker label to each Whisper segment by maximum time overlap.

    Args:
        asr_segments: Whisper segment dicts with keys start, end, text
                      (timestamps relative to chunk start).
        diar_segments: Diarisation result objects with .start, .end, .speaker
                       (timestamps relative to chunk start).
        offset_s: Absolute start time of the chunk in seconds; added to all
                  timestamps to produce episode-level positions.

    Returns:
        List of Turn dicts with absolute start_s / end_s.
    """
    attributed: list[Turn] = []
    for seg in asr_segments:
        best_speaker = "Unknown"
        best_ov = 0.0
        for diar in diar_segments:
            ov = _overlap(seg["start"], seg["end"], diar.start, diar.end)
            if ov > best_ov:
                best_ov = ov
                best_speaker = f"Speaker {diar.speaker}"
        attributed.append(
            {
                "speaker": best_speaker,
                "text": seg["text"].strip(),
                "start_s": seg["start"] + offset_s,
                "end_s": seg["end"] + offset_s,
            }
        )
    return attributed


def _group_turns(attributed: list[Turn]) -> list[Turn]:
    """Merge consecutive same-speaker segments into turns, keeping timestamps."""
    turns: list[Turn] = []
    for item in attributed:
        if turns and turns[-1]["speaker"] == item["speaker"]:
            turns[-1]["text"] += " " + item["text"]
            turns[-1]["end_s"] = item["end_s"]
        else:
            turns.append(dict(item))
    return turns


def _process_chunk(
    chunk_audio: "AudioSegment",
    chunk_index: int,
    total_chunks: int,
    offset_s: float,
    asr_model,
    diar_model,
    tmp_dir: str,
) -> list[Turn]:
    """Transcribe and diarise one audio chunk, returning attributed turns.

    Args:
        chunk_audio: The audio slice to process.
        chunk_index: 1-based index for logging.
        total_chunks: Total number of chunks for logging.
        offset_s: Absolute start time of this chunk in the episode.
        asr_model: Loaded Whisper ASR model.
        diar_model: Loaded diarisation model.
        tmp_dir: Temporary directory path for intermediate MP3 files.

    Returns:
        List of Turn dicts with episode-level absolute timestamps.
    """
    print(f"  chunk {chunk_index}/{total_chunks}", file=sys.stderr)
    chunk_path = Path(tmp_dir) / f"chunk_{chunk_index:04d}.mp3"
    chunk_audio.export(chunk_path, format="mp3")

    asr_result = generate_transcription(model=asr_model, audio=str(chunk_path))
    diar_result = diar_model.generate(str(chunk_path), threshold=0.5, verbose=False)

    attributed = _assign_speakers(asr_result.segments, diar_result.segments, offset_s)
    return _group_turns(attributed)


# --------------------------------------------------------------------------- #
# Chunk boundary reconciliation
# --------------------------------------------------------------------------- #

def _format_turns(turns: list[Turn]) -> str:
    return "\n".join(f"**{t['speaker']}**: {t['text']}" for t in turns)


def _parse_reconciliation(response: str) -> tuple[list[Turn], dict[str, str]]:
    """Parse the LLM reconciliation response into turns and a speaker mapping.

    Returns:
        (reconciled_turns, mapping) where mapping is {next_label: prev_label}.
    """
    # Split on section headers; be lenient about whitespace.
    turns_block = ""
    mapping_block = ""

    turns_match = re.search(
        r"##\s*RECONCILED TURNS\s*\n(.*?)(?=##\s*SPEAKER MAPPING|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    mapping_match = re.search(
        r"##\s*SPEAKER MAPPING\s*\n(.*?)$",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if turns_match:
        turns_block = turns_match.group(1).strip()
    if mapping_match:
        mapping_block = mapping_match.group(1).strip()

    # Parse turns: **Speaker X**: text
    reconciled: list[Turn] = []
    for line in turns_block.splitlines():
        m = re.match(r"\*\*(.+?)\*\*:\s*(.*)", line.strip())
        if m:
            reconciled.append(
                {"speaker": m.group(1), "text": m.group(2), "start_s": 0.0, "end_s": 0.0}
            )

    # Parse mapping: Next label → Prev label
    mapping: dict[str, str] = {}
    for line in mapping_block.splitlines():
        m = re.match(r"(.+?)\s*[-→>]+\s*(.+)", line.strip())
        if m:
            mapping[m.group(1).strip()] = m.group(2).strip()

    return reconciled, mapping


def _apply_speaker_mapping(turns: list[Turn], mapping: dict[str, str]) -> list[Turn]:
    """Return a copy of turns with speaker labels renamed according to mapping."""
    result = []
    for t in turns:
        new_speaker = mapping.get(t["speaker"], t["speaker"])
        result.append({**t, "speaker": new_speaker})
    return result


def _reconcile_boundary(
    tail: list[Turn],
    head: list[Turn],
    llm,
    tokenizer,
) -> tuple[list[Turn], dict[str, str]]:
    """Ask the LLM to reconcile the overlapping boundary region.

    Args:
        tail: Last BOUNDARY_TURNS turns from the accumulated transcript.
        head: First BOUNDARY_TURNS turns from the incoming chunk.
        llm: Loaded language model.
        tokenizer: Corresponding tokenizer.

    Returns:
        (reconciled_turns, speaker_mapping_for_incoming_chunk)
    """
    prompt_text = _RECONCILE_PROMPT.format(
        tail=_format_turns(tail),
        head=_format_turns(head),
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(llm, tokenizer, prompt=prompt, max_tokens=800, verbose=False)
    return _parse_reconciliation(response)


# --------------------------------------------------------------------------- #
# Full transcription + diarisation pipeline
# --------------------------------------------------------------------------- #

def _transcribe_and_attribute(
    audio_path: Path,
    asr_model_id: str,
    diar_model_id: str,
    llm_model_id: str,
) -> list[Turn]:
    """Run the full ASR → diarisation → attribution → reconciliation pipeline.

    Audio is split into overlapping chunks. Each chunk is independently
    transcribed and diarised. The LLM then iteratively reconciles each chunk
    boundary, deduplicating the overlap and normalising speaker labels.

    Returns:
        Episode-level list of attributed turns.
    """
    audio = AudioSegment.from_file(audio_path)
    duration_s = len(audio) / 1000

    # Build chunk start positions using the non-overlapping stride.
    starts_ms = list(range(0, len(audio), STRIDE_MS))
    total = len(starts_ms)
    print(
        f"Audio: {duration_s:.0f}s → {total} chunk(s) "
        f"(stride={STRIDE_MS//1000}s, overlap={OVERLAP_MS//1000}s)",
        file=sys.stderr,
    )

    # --- Load ASR and diarisation models (keep in memory for all chunks). ---
    print(f"Loading ASR model: {asr_model_id}", file=sys.stderr)
    asr_model = load_asr_model(asr_model_id)
    # The mlx-community cache stores only weights + config; load the processor
    # from the canonical OpenAI repo so get_tokenizer() works.
    asr_model._processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3-turbo"
    )

    print(f"Loading diarisation model: {diar_model_id}", file=sys.stderr)
    diar_model = load_diar_model(diar_model_id)

    chunk_turns: list[list[Turn]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, start_ms in enumerate(starts_ms):
            end_ms = start_ms + STRIDE_MS + OVERLAP_MS
            chunk_audio = audio[start_ms:end_ms]
            turns = _process_chunk(
                chunk_audio=chunk_audio,
                chunk_index=idx + 1,
                total_chunks=total,
                offset_s=start_ms / 1000,
                asr_model=asr_model,
                diar_model=diar_model,
                tmp_dir=tmp,
            )
            chunk_turns.append(turns)

    del asr_model, diar_model
    mx.metal.clear_cache()

    if total == 1:
        return chunk_turns[0]

    # --- Reconcile boundaries iteratively. ----------------------------------
    print(f"Loading LLM for reconciliation: {llm_model_id}", file=sys.stderr)
    llm, tokenizer = load(llm_model_id)

    # accumulated holds the clean, speaker-normalised transcript built so far.
    accumulated = chunk_turns[0]

    for idx in range(1, total):
        print(f"  reconciling boundary {idx}/{total - 1}…", file=sys.stderr)
        incoming = chunk_turns[idx]

        tail = accumulated[-BOUNDARY_TURNS:]
        head = incoming[:BOUNDARY_TURNS]

        reconciled, mapping = _reconcile_boundary(tail, head, llm, tokenizer)

        # Apply the speaker mapping to the ENTIRE incoming chunk so labels are
        # consistent with the accumulated transcript.
        normalised_incoming = _apply_speaker_mapping(incoming, mapping)

        # Replace the tail with the reconciled overlap, then append the
        # non-overlapping remainder of the (now re-labelled) incoming chunk.
        accumulated = accumulated[: len(accumulated) - len(tail)]
        accumulated.extend(reconciled)
        accumulated.extend(normalised_incoming[BOUNDARY_TURNS:])

    del llm, tokenizer
    mx.metal.clear_cache()

    return accumulated


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _evaluate(reference: str, attributed: str, llm_model_id: str) -> str:
    """Evaluate the speaker-attributed transcript against a reference."""
    print(f"Loading LLM for evaluation: {llm_model_id}", file=sys.stderr)
    llm, tokenizer = load(llm_model_id)
    prompt_text = _EVAL_PROMPT.format(reference=reference, attributed=attributed)
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("Evaluating…", file=sys.stderr)
    result = generate(llm, tokenizer, prompt=prompt, max_tokens=1500, verbose=False)
    del llm, tokenizer
    mx.metal.clear_cache()
    return result


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an MP3 with speaker diarisation and evaluate the "
            "speaker-attributed transcript against a reference, printing the "
            "evaluation to stdout."
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
        help=f"Whisper ASR model ID (default: {ASR_MODEL_DEFAULT}).",
    )
    parser.add_argument(
        "--diar-model",
        default=DIAR_MODEL_DEFAULT,
        metavar="MODEL_ID",
        help=f"Diarisation model ID (default: {DIAR_MODEL_DEFAULT}).",
    )
    parser.add_argument(
        "--llm-model",
        default=LLM_MODEL_DEFAULT,
        metavar="MODEL_ID",
        help=f"LLM model ID for reconciliation and evaluation (default: {LLM_MODEL_DEFAULT}).",
    )
    parser.add_argument(
        "--save-transcript",
        type=Path,
        metavar="PATH",
        help="If given, write the speaker-attributed transcript to this file.",
    )

    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"Audio file not found: {args.audio}")
    if not args.reference.exists():
        parser.error(f"Reference transcript not found: {args.reference}")

    turns = _transcribe_and_attribute(
        audio_path=args.audio,
        asr_model_id=args.asr_model,
        diar_model_id=args.diar_model,
        llm_model_id=args.llm_model,
    )
    attributed_text = "\n\n".join(f"**{t['speaker']}**: {t['text']}" for t in turns)

    if args.save_transcript:
        args.save_transcript.parent.mkdir(parents=True, exist_ok=True)
        args.save_transcript.write_text(
            f"# Speaker-Attributed Transcript\n\n{attributed_text}\n"
        )
        print(f"Transcript saved to {args.save_transcript}", file=sys.stderr)

    reference = args.reference.read_text()
    evaluation = _evaluate(reference, attributed_text, args.llm_model)

    print(evaluation)


if __name__ == "__main__":
    main()
