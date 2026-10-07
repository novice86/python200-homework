# Task 3: Inside the Boundary -- weatherkit/records.py
from dataclasses import dataclass

from weatherkit import WeatherResponse

@dataclass
class HourlyReading:
    """
    Represents a single hour's weather observation.
    
    Attributes:
        timestamp (str): The time of the reading.
        temperature_c (float): The air temperature measured in degrees Celsius.
        precipitation_mm (float): The amount of precipitation measured in millimeters.
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """
    Converts columnar API weather data into a row-oriented list of hourly observations.

    Args:
        response (WeatherResponse): The validated top-level Pydantic model containing 
            the columnar hourly weather arrays.

    Returns:
        list[HourlyReading]: A list of parsed readings, ordered chronologically, 
            where each object represents a single hour's combined weather metrics.
    """
    readings = []
    
    # We can safely zip these arrays together because our Pydantic model_validator 
    # previously guaranteed that all three lists are exactly the same length.
    for time_val, temp_val, precip_val in zip(
        response.hourly.time, 
        response.hourly.temperature_2m, 
        response.hourly.precipitation
    ):
        readings.append(
            HourlyReading(
                timestamp=time_val,
                temperature_c=temp_val,
                precipitation_mm=precip_val
            )
        )
        
    return readings

# Why is HourlyReading a dataclass rather than a Pydantic model?
# Pydantic models belong at the "boundary" of the application. Their job is to take 
# messy, untrusted external data (like JSON from a web API) and aggressively parse, 
# cast, and validate it. 
# 
# Once the data successfully passes through that Pydantic boundary (in this case, 
# via WeatherResponse), it becomes trusted internal data. Inside application's 
# core logic, we no longer need the heavy computational overhead of Pydantic's 
# validation engine. A standard Python dataclass is much lighter and faster, making 
# it the correct tool for modeling trusted data inside the boundary.