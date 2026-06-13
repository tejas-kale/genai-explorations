from unittest.mock import patch

import pytest

from agentic_loop import SYSTEM_PROMPT, ask, chat

FIXED_WEATHER = (
    "Weather data for TestCity:\n"
    "Temp: 20°C (feels like 18°C)\n"
    "Condition: Sunny\n"
    "Wind: 10 km/h N\n"
    "Humidity: 45%"
)


def _messages(query):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


WEATHER_QUERIES = [
    "Weather in London?",
    "Rain in Paris?",
    "How hot in Tokyo?",
    "Max temp Berlin?",
    "Umbrella Berlin?",
]

NON_WEATHER_QUERIES = [
    "What is 2+2?",
    "Who wrote Hamlet?",
    "Tell me a joke",
    "How do I cook pasta?",
    "Capital of France?",
    "What time is it?",
    "What's the temperature?",
]


@pytest.mark.parametrize("query", WEATHER_QUERIES)
def test_weather_query_outputs_weathercheck(query):
    response = ask(_messages(query))
    assert "WEATHERCHECK" in response, f"Expected WEATHERCHECK for '{query}', got: {response}"


@pytest.mark.parametrize("query", NON_WEATHER_QUERIES)
def test_non_weather_query_no_weathercheck(query):
    response = ask(_messages(query))
    assert "WEATHERCHECK" not in response, f"Unexpected WEATHERCHECK for '{query}', got: {response}"


@pytest.mark.parametrize("query", WEATHER_QUERIES)
@patch("agentic_loop.fetch_weather", return_value=FIXED_WEATHER)
def test_chat_weather_uses_tool(mock_fetch, query):
    response = chat(query)
    assert "WEATHERCHECK" not in response, f"Loop didn't complete for '{query}'"
    assert "°C" in response, f"Expected temperature data in response for '{query}'"
    mock_fetch.assert_called_once()


@pytest.mark.parametrize("query", NON_WEATHER_QUERIES)
def test_chat_non_weather_single_turn(query):
    response = chat(query)
    assert "WEATHERCHECK" not in response, f"Unexpected WEATHERCHECK for '{query}'"
