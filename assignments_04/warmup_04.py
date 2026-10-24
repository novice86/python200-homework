import json
import pytest
from dataclasses import dataclass

# Warmup Question 1
class ModelInfo:
    def __init__(self, path):
        with open(path, "r") as file:
            self.data = json.load(file)

    def feature_count(self) -> int: 
        return len(self.data["features"])


mock_data = {
    "features": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ]
}

file_name = "test_features.json"
with open(file_name, "w") as file:
    json.dump(mock_data, file, indent=4)

model_info = ModelInfo(file_name)
print(f"Feature count is: {model_info.feature_count()}")
# Loading the file in __init__ caches the data during object initialization, 
# preventing redundant and expensive file reads if feature_count() is called multiple times.

# Warmup Question 2
@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        return f"{self.date}: high {self.high_c}C, " + ("rain expected" if self.will_rain else "rain is not expected")

forecast_no_rain = DayForecast("2023-06-15", 25, False)
print(forecast_no_rain.summary())
forecast_rain = DayForecast("2023-06-16", 16, True)
print(forecast_rain.summary())
# The @dataclass decorator automatically generated the __init__ method.

# Warmup Question 3
def parse_temperature(text: str) -> float:
    try:
        temp = float(text)
        return temp
    except Exception as e:
        raise ValueError(f"Input temperature={text} is not a number")


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")
# pytest.raises(ValueError) without match fail to catch that ValueError occur due to this specific reason that match parameter point to.
