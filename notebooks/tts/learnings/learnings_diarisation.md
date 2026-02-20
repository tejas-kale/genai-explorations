# Learnings: ASR + Diarisation on Apple Silicon (mlx-audio)

**Date**: 20 February 2026  
**Notebook**: `04_exploring_diarisation_mlx_studio.ipynb`

---

## 1. Diarisation with `mlx-community/diar_sortformer_4spk-v1-fp32`

- Load with `mlx_audio.vad.load`, not `mlx_audio.stt`.
- Run inference with `model.generate(path, threshold=0.5, verbose=True)`.
- Output is a `DiarizationOutput` object; segments are accessed as `result.segments`, each being a `DiarizationSegment` object with **attribute** access: `.speaker`, `.start`, `.end`.
- The model supports up to 4 speakers and expects 16kHz mono audio. MP3 input works fine.

---

## 2. Qwen3-ASR does not return usable timestamps on Apple Silicon

- `mlx-community/Qwen3-ASR-1.7B-8bit` via `mlx_audio.stt` returns a **single segment** covering the entire audio (`start=0.0, end=<total duration>`), with no intra-segment timestamps.
- The official Qwen3-ASR API supports word/segment timestamps via `Qwen3-ForcedAligner-0.6B`, but **this model has no MLX port** and only runs on CUDA.
- Therefore, time-overlap alignment between ASR segments and diarisation segments is **not possible** with Qwen3-ASR on Apple Silicon.

---

## 3. ASR segment format: dict vs object

- **Whisper** (via `mlx_audio`) returns segments as **dicts**: `seg["start"]`, `seg["end"]`, `seg["text"]`.
- **Diarisation** segments are **objects**: `seg.start`, `seg.end`, `seg.speaker`.
- Mixing attribute and key access causes `AttributeError: 'dict' object has no attribute 'start'`. Always check the type before writing alignment code.

---

## 4. The correct ASR model for speaker attribution on Apple Silicon: Whisper

Use `mlx-community/whisper-large-v3-turbo`:
- Natively returns per-segment timestamps (`start`, `end`, `text`) as dicts.
- Fast and accurate on M-series chips.
- Compatible with the time-overlap alignment approach.

```python
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model as load_asr

asr_model = load_asr("mlx-community/whisper-large-v3-turbo")
asr_result = generate_transcription(model=asr_model, audio="audio.mp3")
# asr_result.segments → list of dicts with "start", "end", "text"
```

---

## 5. Time-overlap alignment: joining ASR and diarisation

For each Whisper segment, find the diarisation segment with the greatest time overlap and assign that speaker label. Then merge consecutive same-speaker segments into turns.

```python
def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def assign_speakers(asr_segments, diar_segments):
    attributed = []
    for seg in asr_segments:
        best_speaker, best_overlap = "Unknown", 0.0
        for diar in diar_segments:
            ov = _overlap(seg["start"], seg["end"], diar.start, diar.end)
            if ov > best_overlap:
                best_overlap, best_speaker = ov, f"Speaker {diar.speaker}"
        attributed.append({"speaker": best_speaker, "text": seg["text"].strip()})
    return attributed

def group_turns(attributed):
    turns = []
    for item in attributed:
        if turns and turns[-1]["speaker"] == item["speaker"]:
            turns[-1]["text"] += " " + item["text"]
        else:
            turns.append({"speaker": item["speaker"], "text": item["text"]})
    return turns
```

---

## 6. Using an LLM for attribution is not viable on 8 GB Apple Silicon

- The fallback of using an LLM (e.g. `Qwen2.5-7B-Instruct-4bit`) to split the raw transcript according to the diarisation timeline crashed the kernel on an 8 GB M1 MacBook Air.
- A 1.5B model was tried but **hallucinated** speaker names (introduced named speakers like "Arang Kashavazian" instead of using the diarisation labels).
- **Verdict**: LLM-based attribution is not reliable at small model sizes and not feasible at larger sizes on low-memory devices. Timestamp-based alignment is the correct approach.

---

## 7. Model size guidance for Apple Silicon

| Task | Model | VRAM required | 8 GB M1 |
|---|---|---|---|
| ASR (Whisper) | `whisper-large-v3-turbo` | ~1.5 GB | ✅ |
| ASR (Qwen3) | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB | ✅ (but no timestamps) |
| Diarisation | `diar_sortformer_4spk-v1-fp32` | ~0.5 GB | ✅ |
| Evaluation LLM | `Qwen2.5-1.5B-Instruct-4bit` | ~1 GB | ✅ |
| Attribution LLM | `Qwen2.5-3B-Instruct-4bit` | ~2 GB | ✅ (marginal) |
| Attribution LLM | `Qwen2.5-7B-Instruct-4bit` | ~4.5 GB | ⚠️ risky, may crash |

---

## 8. `mx.metal.clear_cache()` — correct invocation

Always use `mx.metal.clear_cache()` after deleting a model to release Metal GPU memory. `mx.clear_cache()` does not exist and will raise `AttributeError`.

---

## 9. Pipeline summary

```
MP3
 ├─ whisper-large-v3-turbo  → segments[(start, end, text), ...]
 └─ diar_sortformer_4spk    → segments[(speaker, start, end), ...]
          │
          ▼ time-overlap assign_speakers()
          │
    attributed [(speaker, text), ...]
          │
          ▼ group_turns()
          │
    turns [(speaker, full turn text), ...]
          │
          ▼ Qwen2.5-1.5B-Instruct-4bit
          │
    evaluation (speaker count, consistency, turn-taking, text-speaker alignment)
```
