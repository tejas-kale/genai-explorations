# Qwen3.5-0.8B Weather Agent

A minimal agentic loop built on `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`: the model
either answers directly or requests live weather data via `WEATHERCHECK <city>`,
which is fetched from `wttr.in` and fed back for a final answer.

## Files
- `agentic_loop.py` — system prompt, model loading, `ask`/`fetch_weather`/`chat`
- `test_agentic_loop.py` — pytest tests for `chat`, mocking the weather API

## Run
```bash
python -m pytest test_agentic_loop.py
```

See `NOTES.md` for the full experiment log (prompt iterations, ablation study,
and why few-shot examples were the fix that made classification reliable).
