import json
import re
import time
import click
import requests
from mlx_lm import load, generate

MODEL_ID = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
MAX_ITERATIONS = 3

SYSTEM_PROMPT = """You are a general-purpose assistant. Answer questions directly.

You may request weather data ONLY when BOTH are true:
  (a) The user mentions a real city by name, AND
  (b) The user asks about weather/rain/sun/temperature/forecast.

When both are true, output exactly: WEATHERCHECK <city>
You will receive weather data next. Then answer the question using it.

When you receive "Weather data for": do NOT output WEATHERCHECK again. Just answer.

Examples of questions you answer directly:
  "Tell me a joke" "What is 2+2?" "Who wrote Hamlet?" "How do I cook pasta?"

Example of weather request:
  "What's the weather in London?" → WEATHERCHECK London"""


def get_weather(city):
    url = f"https://wttr.in/{city}?format=j1"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    today = data["weather"][0]
    current = data["current_condition"][0]
    return (
        f"Condition: {current['weatherDesc'][0]['value']}, "
        f"Temperature: {current['temp_C']}°C (feels like {current['FeelsLikeC']}°C), "
        f"Max today: {today['maxtempC']}°C, Min today: {today['mintempC']}°C, "
        f"Wind: {current['winddir16Point']} {current['windspeedKmph']}km/h, "
        f"Humidity: {current['humidity']}%"
    )


def call_model(model, tokenizer, messages, verbose=False, enable_thinking=False):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not enable_thinking:
        kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    if verbose:
        click.echo(f"\n  [DEBUG] Messages: {len(messages)}")
        for m in messages:
            click.echo(f"  [DEBUG]   [{m['role']}] {m['content'][:150]}")
        click.echo(f"  [DEBUG] Prompt ({len(prompt)} chars): {prompt[:300]}...")
    t0 = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
    elapsed = time.time() - t0
    if verbose:
        click.echo(f"  [DEBUG] Took {elapsed:.1f}s  Response ({len(response)} chars): {repr(response[:200])}")
    return response


@click.command()
@click.argument("prompt")
@click.option("--think/--no-think", default=False, help="Enable thinking mode")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def main(prompt, verbose, think):
    click.echo(f"Loading model {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)
    click.echo("Ready.\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for i in range(MAX_ITERATIONS):
        click.echo(f"--- Iteration {i + 1} ---")
        response = call_model(model, tokenizer, messages, verbose=verbose, enable_thinking=think)
        click.echo(f"Model: {response}")

        match = re.search(r"WEATHERCHECK\s+([A-Z][a-zA-Z\s]+)$", response, re.MULTILINE)
        if match:
            city = match.group(1).strip()
            click.echo(f"\nFetching weather for: {city}")
            weather = get_weather(city)
            click.echo(f"Weather: {weather}")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Weather data for {city}: {weather}"})
        else:
            click.echo("\nDone.")
            break
    else:
        click.echo("\nMax iterations reached.")


if __name__ == "__main__":
    main()
