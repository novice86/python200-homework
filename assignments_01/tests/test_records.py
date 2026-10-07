import pytest
from weatherkit import WeatherResponse, HourlyBlock, to_readings, HourlyReading

@pytest.fixture
def mock_response():
    return WeatherResponse(
        latitude=35.73,
        longitude=-78.83,
        timezone="America/Los_Angeles",
        hourly=HourlyBlock(
            time=["2026-10-01T00:00", "2026-10-01T01:00", "2026-10-01T02:00"],
            temperature_2m=[65.5, 64.2, 63.8],
            precipitation=[0.0, 0.1, 0.5]
        )
    )

def test_to_readings_returns_ordered_hourly_readings(mock_response):
    """Ensures order of readings is preserved"""

    readings = to_readings(mock_response)

    assert len(readings) == 3
    assert readings[0].timestamp == "2026-10-01T00:00"
    assert readings[2].timestamp == "2026-10-01T02:00"


# Parametrize injects these values directly into the test arguments
@pytest.mark.parametrize("index, expected_time, expected_temp, expected_precip", [
    (0, "2026-10-01T00:00", 65.5, 0.0),
    (1, "2026-10-01T01:00", 64.2, 0.1),
    (2, "2026-10-01T02:00", 63.8, 0.5),
])
def test_to_readings_values_match_input_indices(
    mock_response, index, expected_time, expected_temp, expected_precip
):
    """Catches zipping errors by ensuring values aren't paired with the wrong timestamps."""
    readings = to_readings(mock_response)
    reading = readings[index]
    
    assert reading.timestamp == expected_time
    assert reading.temperature_c == expected_temp
    assert reading.precipitation_mm == expected_precip


def test_identical_hourly_reading_objects_compare_equal():
    """Ensures two distinct instances with the same data compare as equal (==)."""
    reading_1 = HourlyReading(
        timestamp="2026-10-01T00:00", 
        temperature_c=65.5, 
        precipitation_mm=0.0
    )
    
    reading_2 = HourlyReading(
        timestamp="2026-10-01T00:00", 
        temperature_c=65.5, 
        precipitation_mm=0.0
    )
    
    assert reading_1 == reading_2
    assert reading_1 is not reading_2