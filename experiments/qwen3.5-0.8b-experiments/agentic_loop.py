import re

import mlx_lm
import requests
from transformers import AutoTokenizer

MODEL_ID = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"

SYSTEM_PROMPT = """\
You are a general-purpose assistant. Answer questions directly.

You may request weather data ONLY when BOTH are true:
  (a) The user mentions a real city by name, AND
  (b) The user asks about weather/rain/sun/temperature/forecast.

When both are true, output exactly: WEATHERCHECK <city>
Otherwise, just answer the question.

When you receive weather data, do NOT output WEATHERCHECK again. Just answer using it."""

model, tokenizer = mlx_lm.load(MODEL_ID)


def ask(messages):
    prompt = tokenizer.apply_chat_template(
        messages, enable_thinking=False, add_generation_prompt=True
    )
    return mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=256).strip()


def fetch_weather(city):
    url = f"https://wttr.in/{city}?format=j1"
    data = requests.get(url, timeout=10).json()
    cc = data["current_condition"][0]
    return (
        f"Weather data for {city}:\n"
        f"Temp: {cc['temp_C']}°C (feels like {cc['FeelsLikeC']}°C)\n"
        f"Condition: {cc['weatherDesc'][0]['value']}\n"
        f"Wind: {cc['windspeedKmph']} km/h {cc['winddir16Point']}\n"
        f"Humidity: {cc['humidity']}%"
    )


def chat(user_msg, max_turns=3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    for _ in range(max_turns):
        response = ask(messages)
        match = re.search(r"WEATHERCHECK\s+(.+)", response)
        if not match:
            return response
        city = match.group(1).strip()
        weather = fetch_weather(city)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": weather})
    return response
