import json
import pytest
from pydantic import ValidationError
from weatherkit import WeatherResponse
from pathlib import Path


WEATHER_DATA_PATH = Path(__file__).parent.parent / "weather_raw.json"


def test_valid_real_data_loads_and_has_168_hours():
    """
    Test that the real weather data can be loaded and validated, and that it contains exactly 168 hourly readings.
    """
    with open(WEATHER_DATA_PATH, "r") as f:
        raw_data = json.load(f)

    # Validate the dictionary using the Pydantic model
    validated_weather = WeatherResponse.model_validate(raw_data)

    # Check that there are exactly 168 hourly readings
    assert len(validated_weather.hourly.time) == 168


def test_invalid_latitude_raises_validation_error():
    """
    Test that an invalid latitude value raises a validation error.
    """
    invalid_data = {
        "latitude": 200.0,  # Invalid latitude
        "longitude": -122.4194,
        "timezone": "America/Los_Angeles",
        "hourly": {
            "time": ["2023-01-01T00:00:00Z"],
            "temperature_2m": [10.0],
            "precipitation": [0.0]
        }
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(invalid_data)

def test_mismatched_hourly_lengths_raises_validation_error():
    """
    Test that mismatched lengths of hourly data arrays raise a validation error.
    """
    invalid_data = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timezone": "America/Los_Angeles",
        "hourly": {
            "time": ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z"],
            "temperature_2m": [10.0],  # Mismatched length
            "precipitation": [0.0, 0.1]
        }
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(invalid_data)


def test_null_in_temperature_array_raises_error():
    """Ensures the schema rejects null (None) values inside numerical arrays."""
    invalid_data = {
        "latitude": 35.73,
        "longitude": -78.83,
        "hourly": {
            "time": ["2026-10-01T00:00", "2026-10-01T01:00", "2026-10-01T02:00"],
            "temperature_2m": [65.5, None, 63.8],  # Contains an invalid None
            "precipitation": [0.0, 0.0, 0.0]
        }
    }
    
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(invalid_data)