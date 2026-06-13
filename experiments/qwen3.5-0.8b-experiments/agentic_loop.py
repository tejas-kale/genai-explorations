import re

import mlx_lm
import requests
from transformers import AutoTokenizer

MODEL_ID = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"

SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question directly.

If the user asks about weather in a specific city, output: WEATHERCHECK <city>
Otherwise, just answer the question.

Example 1:
User: Weather in London?
Assistant: WEATHERCHECK London

Example 2:
User: What is 2+2?
Assistant: 4

Example 3:
User: Who wrote Hamlet?
Assistant: William Shakespeare

Example 4:
User: Rain in Paris?
Assistant: WEATHERCHECK Paris

Example 5:
User: Tell me a joke
Assistant: Why don't scientists trust atoms? Because they make up everything!

Example 6:
User: How do I cook pasta?
Assistant: Boil salted water, add pasta, cook for 8-10 minutes until al dente.

Example 7:
User: Capital of France?
Assistant: Paris

Example 8:
User: What time is it?
Assistant: I don't have access to the current time.

Example 9:
User: What's the temperature?
Assistant: I need a city name to check the temperature.

Example 10:
User: How hot in Tokyo?
Assistant: WEATHERCHECK Tokyo

Example 11:
User: Max temp Berlin?
Assistant: WEATHERCHECK Berlin

Example 12:
User: Umbrella Berlin?
Assistant: WEATHERCHECK Berlin

When you receive weather data, answer using it. Do NOT output WEATHERCHECK again.

Example (full flow):
User: Weather in Tokyo?
Assistant: WEATHERCHECK Tokyo
User: Weather data for Tokyo:
Temp: 25°C (feels like 27°C)
Condition: Partly cloudy
Wind: 8 km/h SE
Humidity: 60%
Assistant: It's 25°C in Tokyo, feels like 27°C. Partly cloudy with 60% humidity."""

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
