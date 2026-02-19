# Learnings: German Podcast → English Pipeline (pyannote 4.x + transformers + TranslateGemma on PyTorch 2.8 Stable)

## Environment

- PyTorch 2.8.0+cu128 (stable release)
- Python 3.12
- pyannote.audio 4.0.4
- transformers 5.x
- CUDA 12.8 / RTX PRO 6000

---

## 1. WhisperX requires CTranslate2-format models

WhisperX uses `faster-whisper` under the hood, which requires models converted to **CTranslate2 format** (`model.bin`). Standard HuggingFace Whisper fine-tunes (e.g. `primeline/whisper-large-v3-german`) store weights as `.safetensors` or `pytorch_model.bin` and will fail with `Unable to open file 'model.bin'`.

**Fix**: Use the native transformers `pipeline("automatic-speech-recognition")` with `AutoModelForSpeechSeq2Seq`, which loads `.safetensors` directly.

---

## 2. torchcodec is incompatible with PyTorch 2.8.0 stable

`torchcodec` ships its own internal shared library (`libtorchcodec_coreN.so`) that calls `aoti_torch_create_device_guard`. This symbol exists only in **nightly PyTorch** builds — it is absent from the PyTorch 2.8.0 stable release.

Consequences:
- The stable `torchcodec` PyPI wheel fails on import.
- The nightly `torchcodec` wheel from `https://download.pytorch.org/whl/nightly/cu128` also fails, because it was built against a nightly PyTorch that has the symbol. The stable 2.8.0 does not.
- There is no working torchcodec wheel for PyTorch 2.8.0 stable.

**Additional trap**: `pip uninstall torchcodec` removes the package metadata (`.dist-info`) but leaves the module files in `site-packages`. `importlib.util.find_spec` finds packages by file presence, not metadata, so `is_torchcodec_available()` in transformers continues to return `True` after uninstall. The broken import will keep crashing until the directory is manually deleted:
```bash
rm -rf .venv/lib/python3.12/site-packages/torchcodec*
```

---

## 3. pyannote.audio model versions are tightly coupled to the library version

`pyannote/speaker-diarization-community-1` was released alongside pyannote.audio 4.0. Its `config.yaml` passes a `plda` initialisation argument to `SpeakerDiarization.__init__()`, a parameter that only exists in 4.x. Loading this model with pyannote.audio 3.x raises:

```
TypeError: SpeakerDiarization.__init__() got an unexpected keyword argument 'plda'
```

Downgrading to pyannote.audio 3.x to avoid the torchcodec dependency is therefore not viable for this model.

---

## 4. pyannote.audio 4.x renamed the authentication parameter

| Version | Parameter |
|---------|-----------|
| ≤ 3.x | `use_auth_token=` |
| ≥ 4.0 | `token=` |

---

## 5. Both transformers and pyannote.audio 4.x can bypass torchcodec via pre-loaded arrays

Both libraries only invoke torchcodec when given a **file path**. Passing pre-loaded audio avoids the dependency entirely:

**transformers ASR pipeline** — pass a dict:
```python
audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
result = asr_pipe({"raw": audio_array, "sampling_rate": 16000}, ...)
```

**pyannote.audio 4.x pipeline** — pass a waveform tensor:
```python
waveform = torch.tensor(audio_array).unsqueeze(0)  # (1, samples)
diarization = diarize_pipeline({"waveform": waveform, "sample_rate": 16000})
```

This approach works regardless of torchcodec's install state, making it robust to the version incompatibility above.

---

## 6. pyannote.audio 4.x changed the diarization output iteration API

In 3.x the pipeline returned an `Annotation` object; in 4.x it returns a `DiarizeOutput`. The iteration interface is different:

| Version | Code |
|---------|------|
| ≤ 3.x | `for turn, _, speaker in diarization.itertracks(yield_label=True):` |
| ≥ 4.0 | `for turn, speaker in diarization.speaker_diarization:` |

Calling `.itertracks()` on a `DiarizeOutput` raises:
```
AttributeError: 'DiarizeOutput' object has no attribute 'itertracks'
```

**Fix**: replace the `itertracks` loop with iteration over `.speaker_diarization`, which yields `(turn, speaker)` 2-tuples instead of 3-tuples.

---

## 7. Instruction-tuned translation models require the chat template

`google/translategemma-12b-it` (and any `-it` / instruction-tuned model) expects its special control tokens (`<start_of_turn>user`, `<start_of_turn>model`, etc.) to be inserted around the prompt. Passing a raw string skips these tokens, producing garbage logits with NaN/`-inf` probabilities.

When `torch.multinomial` samples from a probability distribution containing NaN it triggers a CUDA device-side assertion:
```
AcceleratorError: CUDA error: device-side assert triggered
```

**Fix**: use `tokenizer.apply_chat_template()` with `do_sample=False` (greedy, deterministic, avoids `multinomial` entirely). However, `translategemma-12b-it` has a **custom Jinja template** that additionally requires `content` to be a list of one structured dict — not a plain string:

```python
messages = [{"role": "user", "content": [{
    "type": "text",
    "source_lang_code": "de",
    "target_lang_code": "en",
    "text": text,
}]}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
).to("cuda")
outputs = translate_model.generate(**inputs, max_new_tokens=512, do_sample=False)
```

Passing a plain string as `content` raises:
```
TemplateError: User role must provide `content` as an iterable with exactly one item.
That item must be a `mapping(type:'text'|'image', source_lang_code:string, target_lang_code:string, ...)`.
```

Note: with `device_map="auto"` the model is sharded across devices — use `.to("cuda")` (first CUDA device) rather than `.to(translate_model.device)` for the inputs.

---

## 8. vLLM requires HF_TOKEN in the environment for gated models

`google/gemma-3-4b-it` is a gated model on Hugging Face. vLLM does not accept a `token=` parameter directly; it reads credentials from the environment variable `HF_TOKEN`. Set it **before** constructing the `LLM` object:

```python
import os
os.environ["HF_TOKEN"] = HF_TOKEN
llm = LLM(model="google/gemma-3-4b-it", ...)
```

---

## 9. vLLM does not release GPU memory until the LLM object is fully garbage-collected

Calling `del llm` alone may not immediately free VRAM because Python's garbage collector is not deterministic. Force collection before loading the next model:

```python
import gc
del llm
gc.collect()
torch.cuda.empty_cache()
```

---

## Summary

The combination of pyannote.audio 4.x + transformers on PyTorch 2.8.0 stable creates a torchcodec dead-end with no installable wheel. The correct workaround is to pre-load audio with librosa and feed raw arrays/tensors to both pipelines, bypassing the torchcodec code path in both libraries completely.

For instruction-tuned translation models (e.g. TranslateGemma, Gemma 3), always apply the tokenizer's chat template — raw prompts silently skip required control tokens, causing NaN probabilities and CUDA assertion crashes.

For fast batch translation, use vLLM instead of the HuggingFace `generate()` loop. vLLM's continuous batching and PagedAttention process all segments in a single pass, reducing translation of a 2-hour podcast from ~10 minutes to under 1 minute on an RTX PRO 6000.
