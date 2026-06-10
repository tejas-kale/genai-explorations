import re

import requests
from IPython.display import Markdown
from mlx_lm import generate, load

MODEL_ID = ...
QUERIES = ...
QUICK_QUERIES = ...
STAGES = ...
N_RUNS = ...


def load_model():
    ...


def chat(tokenizer, messages):
    ...


def answer(model, tokenizer, messages):
    ...


def weather(city):
    ...


def agent_once(model, tokenizer, prompt, regex, query):
    ...


def trial(model, tokenizer, prompt, regex, query, weather_query):
    ...


def score(model, tokenizer, stage, queries, n):
    ...


def table(results):
    ...


def main():
    model, tokenizer = load_model()

    MODEL_ID

    messages = [{"role": "user", "content": "What is 2+2?"}]
    answer(model, tokenizer, messages)

    weather("Berlin")

    prompt, regex = STAGES["base"]
    agent_once(model, tokenizer, prompt, regex, "What is the weather in London?")

    prompt, regex = STAGES["base"]
    agent_once(model, tokenizer, prompt, regex, "What is the capital of France?")

    eval_queries = QUICK_QUERIES
    n_runs = N_RUNS
    len(eval_queries), n_runs

    results = {
        name: score(model, tokenizer, stage, eval_queries, n_runs)
        for name, stage in STAGES.items()
    }

    Markdown(table(results))


if __name__ == "__main__":
    main()
