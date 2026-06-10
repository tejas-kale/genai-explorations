# Qwen3.5-0.8B Experiments

## Setup
- **Model**: `mlx-community/Qwen3.5-0.8B-OptiQ-4bit` (4-bit MLX quant, 0.6 GB on disk)
- **Backend**: MLX (switched from Ollama — Ollama's Qwen3.5 renderer always outputs thinking tokens and `think: false` option was ineffective)
- **Thinking**: Disabled via `tokenizer.apply_chat_template(..., enable_thinking=False)`. This injects empty ` thinking\n\n response` tags into the prompt, which suppresses reasoning output.
- **Agent loop**: User query → model responds (either `[WEATHER: city]` or direct answer) → if weather tag found, fetch from wttr.in JSON API → feed back → model answers

## Experiment 1: Initial Ollama (2B → 0.8B port)
- Replaced `qwen3.5:2b` with `qwen3.5:0.8b` in `agent.py`, renamed folder
- **Result**: Model worked but was very slow (0.8B spends all tokens on thinking, `done_reason: length`)
- **Fix**: Increased `num_predict` to 2048, added fallback to `thinking` field when `content` is empty

## Experiment 2: System prompt v1
- Prompt: "You are a weather assistant. When asked for weather, output [WEATHER: city_name]..."
- **Result**: Model extracted temperature instead of city name: `[WEATHER: 23°C]`
- **Fix**: Made system prompt more explicit with examples and wrong answers

## Experiment 3: Ollama `enable_thinking: false`
- Tried passing `enable_thinking: false` in Ollama options
- **Result**: Did NOT work — Ollama's Qwen3.5 renderer ignores it, kept producing thinking tokens
- **User corrected**: The correct Ollama option is `think: false` (not `enable_thinking`)
- **But**: Even `think: false` was unreliable in Ollama

## Experiment 4: Switch to MLX
- Switched from Ollama to `mlx-lm` with `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`
- **Why**: MLX gives us direct control over the chat template. `enable_thinking=False` in `apply_chat_template` actually works — it injects empty think tags.
- **Result**: Clean, fast output (~1s per turn), no thinking tokens
- **Weather data upgraded**: Switched from `wttr.in?format=%C+%t+%w+%h` (single-line current only) to `wttr.in?format=j1` (JSON API) — now includes max/min temps, feels-like, wind direction, humidity

## Experiment 5: System prompt v2 — distinguishing weather vs non-weather
- Enhanced system prompt with explicit examples, step-by-step decision logic
- **Results across prompt iterations** (single-run, binary ticks):

| Query | v1 (basic) | v2 (examples) | v3 (checklist) | v4 (flowchart) |
|-------|-----------|---------------|----------------|----------------|
| "Tell me a joke" | ❌ [WEATHER: Munich] | ❌ [WEATHER: Paris] (loop) | ❌ [WEATHER: Tokyo] (loop) | ❌ [WEATHER: weather] (loop) |
| "What is 2+2?" | ❌ [WEATHER: Munich] | ✅ 4 | ✅ 4 | ❌ [WEATHER: city_name] |
| "Who wrote Hamlet?" | ❌ [WEATHER: Shakespeare] | ❌ [WEATHER: Shakespeare] | ❌ [WEATHER: Hamlet] | ❌ [WEATHER: London] |
| "How do I cook pasta?" | ❌ [WEATHER: Berlin] | ❌ [WEATHER: Rome] | ❌ [WEATHER: Rome] | ⚠️ [WEATHER: city_name] then answers |
| "Weather in London?" | ✅ correct | ✅ correct | ✅ correct (looped) | ✅ correct (looped) |
| "Rain in Paris?" | ✅ correct | ✅ correct | ✅ correct | ✅ correct |
| "What time is it?" | ❌ [WEATHER: Munich] | ✅ direct | ❌ [WEATHER: Tokyo] | ✅ [TIME: 12:00] |
| "What's the temperature?" | ❌ [WEATHER: Munich] | ⚠️ [WEATHER: New York] | ❌ [WEATHER: Tokyo] | ❌ [WEATHER: London] |

- **Key finding**: The 0.8B model pattern-matches `[WEATHER: ...]` from the system prompt and applies it indiscriminately. No prompt engineering within 4 iterations reliably fixed this.

## Experiment 6: Thinking mode — low/medium vs off
- **Hypothesis**: With thinking enabled, the model might reason about whether a query is weather-related
- **Method**: Added `--think` flag to agent.py. Removed `enable_thinking=False` from chat template (defaults to on). Model generates inside ` thinking...` tags then `</think>` then actual response.
- **Result**: FAILED completely. The 0.8B model never closes ` response` — even with `max_tokens=4096` (16K chars output), it produces endless thinking without ever switching to the actual answer.
- **Verdict**: Thinking mode is unusable for this model size. Stick with `enable_thinking=False`.

## Experiment 7: Breaking the pattern + ablation study
- **Initial approach**: Changed trigger from `[WEATHER: city_name]` to `WEATHERCHECK <city>`, rewrote system prompt. Result appeared to work (single-run ticks). 
- **Problem**: 5 things were changed simultaneously — confounded experiment.
- **Ablation**: Initial harness was subtractive: start with all changes, remove one at a time. This answers "what breaks if I remove X?" It does **not** establish the true baseline.
- **Current notebook rewrite**: `ablation.ipynb` is now a tutorial:
  - load and call the MLX model
  - add the `wttr.in` weather tool
  - show base-prompt failures
  - add prompt changes cumulatively
- **Removed from notebook**: subtractive ablation. Historical subtractive results remain below as notes only.
- **Execution strategy**: Notebook defaults to a cheap smoke test (`QUICK_QUERIES`, `N_RUNS = 1`) so it executes quickly. For evidence, set `eval_queries = QUERIES` and `N_RUNS = 5`.

**Baseline system prompt:**
```
You are a general-purpose assistant. Answer questions directly.

You may request weather data ONLY when BOTH are true:
  (a) The user mentions a real city by name, AND
  (b) The user asks about weather/rain/sun/temperature/forecast.

When both are true, output exactly: WEATHERCHECK <city>
You will receive weather data next. Then answer the question using it.

When you receive "Weather data for": do NOT output WEATHERCHECK again. Just answer.
```

**Variant definitions** (each removes exactly one change from baseline):
- **A**: Revert trigger to `[WEATHER: city]` bracket format
- **B**: Revert regex to lenient `(.+)`
- **C**: Revert framing to "You are a weather assistant"
- **D**: Remove two-condition BOTH gate
- **E**: Remove anti-loop instruction

**Results** (5 runs per query, 12 queries × 6 variants × 5 runs = 360 calls):

| Query | Baseline | A: brackets | B: lenient regex | C: weather asst | D: no gate | E: no anti-loop |
|-------|----------|-------------|------------------|-----------------|------------|-----------------|
| Tell me a joke | 5/5 | 5/5 | 5/5 | **0/5** | 5/5 | 5/5 |
| What is 2+2? | 5/5 | **0/5** | 5/5 | **0/5** | 5/5 | **0/5** |
| Who wrote Hamlet? | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| How do I cook pasta? | 5/5 | **0/5** | 5/5 | **0/5** | 5/5 | **0/5** |
| Capital of France? | 5/5 | **0/5** | 5/5 | **0/5** | **0/5** | **0/5** |
| What time is it? | 5/5 | 5/5 | 5/5 | **0/5** | 5/5 | 5/5 |
| What's the temperature? | 5/5 | 5/5 | 5/5 | **0/5** | 5/5 | **0/5** |
| Weather in London? | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | **0/5** |
| Rain in Paris? | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| How hot in Tokyo? | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Max temp Berlin? | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Umbrella Berlin? | 5/5 | **0/5** | 5/5 | 5/5 | 5/5 | 5/5 |
| **Total** | **60/60** | 40/60 | 60/60 | 30/60 | 55/60 | 35/60 |

**Ablation summary** (ranked by Δ from baseline):

| Rank | Variant | Total | Δ | What it means |
|------|---------|-------|---|---------------|
| — | Baseline | 60/60 | — | — |
| 5 | B: Lenient regex | 60/60 | **0** | **Zero effect.** Model doesn't produce garbage with this prompt. Strict regex is a safety net — no correctness contribution. |
| 4 | D: No two-condition gate | 55/60 | **-5** | **Minor.** Only trips "capital of France"→Paris (the word "France" looks like a location). |
| 3 | A: `[WEATHER: city]` brackets | 40/60 | **-20** | **Significant.** Bracket format is a familiar function-call pattern. The model eagerly matches familiar tool-call syntax. `WEATHERCHECK` is novel and resists over-application. |
| 2 | E: No anti-loop instruction | 35/60 | **-25** | **Major.** Without "do NOT output WEATHERCHECK again", the model doesn't know when to switch from tool-use mode to answer mode. Mode boundaries must be taught explicitly. |
| 1 | C: "Weather assistant" framing | 30/60 | **-30** | **Dominant.** The assistant's IDENTITY is the strongest lever. "Weather assistant" makes the model route everything to weather. "General-purpose assistant" makes weather the rare exception. |

**Corrected explanation — what actually matters, ranked:**
1. **Identity framing** (Δ=-30). The model conforms to its assigned role. If you call it a weather assistant, every query becomes weather. The sentence "Answer questions directly" establishes answering as the default behaviour.
2. **Mode-switching instruction** (Δ=-25). Without an explicit instruction to stop using the tool after receiving data, the model defaults to its most recent behaviour pattern. "Do NOT output WEATHERCHECK again" is the mode-switch trigger.
3. **Novel trigger token** (Δ=-20). Bracket format `[WEATHER: city]` is a familiar function-call pattern from training data. The model over-applies it. A novel token like `WEATHERCHECK` disrupts the pattern-match.
4. **Two-condition gate** (Δ=-5). Marginal. The explicit BOTH rule prevents one edge case but isn't load-bearing.
5. **Regex strictness** (Δ=0). No effect on model behaviour. Could be removed. Included only as defensive engineering.

**Interaction effects** (combination testing):
Single-variable ablation assumes independence. To check: tested key combinations.

| Variant | NW | W | Total |
|---------|-----|-----|-------|
| Baseline (all 5) | 35/35 | 25/25 | 60/60 |
| C only (weather asst) | 5/35 | 25/25 | 30/60 |
| E only (no anti-loop) | 15/35 | 20/25 | 35/60 |
| A only (brackets) | 20/35 | 20/25 | 40/60 |
| **C + E combined** | 15/35 | 20/25 | 35/60 |
| **A + E combined** | 5/35 | 20/25 | 25/60 |
| **C + A combined** | 0/35 | 25/25 | 25/60 |

Findings:
- **C + E = 35/60** (same as E alone). The framing benefit is completely SUBSUMED by the anti-loop removal. Identity framing only helps if you also teach mode-switching. Anti-loop is a prerequisite, not a peer.
- **A + E = 25/60** (worse than either alone). Bracket format + no anti-loop is super-additive damage. The model pattern-matches brackets AND doesn't know when to stop.
- **C + A = 25/60** (worse than either alone). Weather assistant + bracket format = 0/35 on non-weather. The two regressions together destroy classification entirely.

**Key insight from interactions**: The changes are NOT independent. Anti-loop (E) is a gatekeeper — without it, neither framing (C) nor trigger format (A) can function. Bracket format (A) amplifies any other regression multiplicatively. Single-variable ablation reports marginal effects but hides these structural dependencies.

**Methodology correction:**
- The initial Experiment 7 report used single-run binary ticks (✅/❌). Those were misleading:
  - "Capital of France" passes 5/5 in baseline but 0/5 in variants — it has ~20% per-run chance, not a binary property
  - Loop behaviours in single runs masked real variance
  - Single runs cannot distinguish a 20% success rate from a 0% rate
- **Key lesson**: Five variables changed at once → confounded. Any "why it worked" list that includes all five is a post-hoc rationalisation, not an explanation. Ablation separates correlation from causation.

---

## Notebook style rules
- Keep notebook code cells short: ideally ≤5 lines.
- Put all imports in the first code cell.
- Intersperse markdown explanations between code sections.
- Avoid `print`; rely on Jupyter's default display output.
- Put longer helper functions in small `.py` modules with clear names and docstrings.

## Implementation details
- **Weather API**: `https://wttr.in/{city}?format=j1` returns JSON with `current_condition` and `weather` (forecast) arrays
- **Chat template**: Qwen3.5 uses `<|im_start|>role\ncontent<|im_end|>` format
- **Thinking control**: `tokenizer.apply_chat_template(messages, enable_thinking=False)` is the only reliable method
- **Model size**: 873M parameters, 4-bit quantized, ~0.6 GB on disk