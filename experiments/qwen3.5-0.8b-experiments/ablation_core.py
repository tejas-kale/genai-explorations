import re
import requests
from mlx_lm import load, generate

MODEL_ID = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"

QUERIES = [
    ("non-weather", "Tell me a joke"),
    ("non-weather", "What is 2+2?"),
    ("non-weather", "Who wrote Hamlet?"),
    ("non-weather", "How do I cook pasta?"),
    ("non-weather", "What is the capital of France?"),
    ("non-weather", "What time is it?"),
    ("non-weather", "What's the temperature?"),
    ("weather", "What is the weather in London?"),
    ("weather", "Is it going to rain in Paris?"),
    ("weather", "How hot is it in Tokyo right now?"),
    ("weather", "What is the maximum expected temperature in Berlin today?"),
    ("weather", "Should I bring an umbrella to Berlin today?"),
]

QUICK_QUERIES = [
    ("non-weather", "What is 2+2?"),
    ("non-weather", "What is the capital of France?"),
    ("weather", "What is the weather in London?"),
    ("weather", "What is the maximum expected temperature in Berlin today?"),
]

BASE_PROMPT = """You are a weather assistant.

If a user asks about weather conditions for a specific city,
output exactly: [WEATHER: city]
You will receive weather data next.

Convert the weather data into a single natural sentence for the user."""

FRAMED_PROMPT = BASE_PROMPT.replace(
    "You are a weather assistant.",
    "You are a general-purpose assistant. Answer questions directly.",
)

TRIGGER_PROMPT = FRAMED_PROMPT.replace("[WEATHER: city]", "WEATHERCHECK <city>")

GATED_PROMPT = TRIGGER_PROMPT.replace(
    "If a user asks about weather conditions for a specific city,\noutput exactly:",
    "You may request weather data ONLY when BOTH are true:\n"
    "  (a) The user mentions a real city by name, AND\n"
    "  (b) The user asks about weather/rain/sun/temperature/forecast.\n\n"
    "When both are true, output exactly:",
)

FINAL_PROMPT = GATED_PROMPT + '\n\nWhen you receive "Weather data for": do NOT output WEATHERCHECK again. Just answer.'

REGEX_BRACKET = r"\[WEATHER:\s*([A-Z][a-zA-Z\s]+)\]"
REGEX_WEATHERCHECK = r"WEATHERCHECK\s+([A-Z][a-zA-Z\s]+)$"

STAGES = {
    "base": (BASE_PROMPT, REGEX_BRACKET),
    "+ framing": (FRAMED_PROMPT, REGEX_BRACKET),
    "+ trigger": (TRIGGER_PROMPT, REGEX_WEATHERCHECK),
    "+ gate": (GATED_PROMPT, REGEX_WEATHERCHECK),
    "+ anti-loop": (FINAL_PROMPT, REGEX_WEATHERCHECK),
}


def load_model():
    """Load the MLX model and tokenizer."""
    return load(MODEL_ID)


def chat(tokenizer, messages):
    """Render chat messages with Qwen thinking disabled."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def answer(model, tokenizer, messages):
    """Generate one response from a chat transcript."""
    return generate(model, tokenizer, prompt=chat(tokenizer, messages), max_tokens=512)


def weather(city):
    """Return compact weather text for a city."""
    data = requests.get(f"https://wttr.in/{city}?format=j1", timeout=15).json()
    today = data["weather"][0]
    current = data["current_condition"][0]
    return f"Condition: {current['weatherDesc'][0]['value']}, Temperature: {current['temp_C']}C, Max today: {today['maxtempC']}C, Min today: {today['mintempC']}C, Humidity: {current['humidity']}%"


def agent_once(model, tokenizer, prompt, regex, query):
    """Run the first agent turn and return response plus extracted city."""
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": query}]
    response = answer(model, tokenizer, messages)
    match = re.search(regex, response, re.MULTILINE)
    return {"response": response, "city": match.group(1).strip() if match else None}


def trial(model, tokenizer, prompt, regex, query, weather_query):
    """Return whether one agent run routed correctly."""
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": query}]
    used_tool = False
    for _ in range(3):
        response = answer(model, tokenizer, messages)
        match = re.search(regex, response, re.MULTILINE)
        if not match:
            return used_tool or not weather_query
        if not weather_query:
            return False
        city = match.group(1).strip()
        messages += [{"role": "assistant", "content": response}, {"role": "user", "content": f"Weather data for {city}: {weather(city)}"}]
        used_tool = True
    return False


def score(model, tokenizer, stage, queries=QUICK_QUERIES, n=1):
    """Return pass counts for one prompt stage."""
    prompt, regex = stage
    rows = [(kind, q, sum(trial(model, tokenizer, prompt, regex, q, kind == "weather") for _ in range(n))) for kind, q in queries]
    nw = sum(v for kind, _, v in rows if kind == "non-weather")
    w = sum(v for kind, _, v in rows if kind == "weather")
    nw_max = n * sum(kind == "non-weather" for kind, _ in queries)
    w_max = n * sum(kind == "weather" for kind, _ in queries)
    return {"non_weather": f"{nw}/{nw_max}", "weather": f"{w}/{w_max}", "total": f"{nw + w}/{nw_max + w_max}", "rows": rows}


def table(results):
    """Return a markdown score table."""
    lines = ["| Stage | Non-weather | Weather | Total |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['non_weather']} | {v['weather']} | {v['total']} |" for k, v in results.items()]
    return "\n".join(lines)
