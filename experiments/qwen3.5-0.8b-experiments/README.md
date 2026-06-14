# Qwen3.5-0.8B Weather Agent

## About
A minimal agentic loop built on `mlx-community/Qwen3.5-0.8B-OptiQ-4bit` intended to test the tool calling ability of the model. Over 3 turns, the model is asked to either give an answer or make a tool call. 

The tool is a call to `wttr.in` which fetches the live weather data for the input city. The model signals its intention for a tool call by responding with `WEATHERCHECK <city>`.

## Learnings
Here are my learnings about working with small LLMs from the experiment:
1. Making the tool call the exception was important to prevent it from calling the tool with each turn. Hence, the prompt states an AND condition for the tool call.
2. It was equally important to instruct the model in the prompt to not call the tool again once it has received the tool output. It would otherwise request the tool call after every iteration.
3. The model still suffers from unexpected failures which needs further experimentation. For example, it makes the tool call correctly for the question "Is it raining in Paris today?" but fails to do so if the question is shortened to "Rain in Paris?".

## Files
- `agentic_loop.py` — system prompt, model loading, `ask`/`fetch_weather`/`chat`
- `test_agentic_loop.py` — pytest tests for `chat`, mocking the weather API

## Run
```bash
python -m pytest test_agentic_loop.py
```

## Next steps
This experiment only tunes the prompt for a single tool call. Further experiments when the agentic system has multiple tools are needed to make it robust for regular use.
