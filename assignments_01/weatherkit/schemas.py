# Task 2: The Boundary -- weatherkit/schemas.py
from pydantic import BaseModel, Field, model_validator

class HourlyBlock(BaseModel):
    """
    Represents the columnar hourly weather data returned by the API. Contains parallel arrays for
    time, temperature, and precipitation that must all share the same length.
    """

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_list_lengths(self) -> "HourlyBlock":
        """
        Validates that the lengths of the time, temperature, and precipitation lists are equal.
        Raises a ValueError if they are not.
        """
        length_time = len(self.time)
        length_temp = len(self.temperature_2m)
        length_precip = len(self.precipitation)

        if not (length_time == length_temp == length_precip):
            raise ValueError(
                f"List lengths are not equal: time={length_time}, "
                f"temperature_2m={length_temp}, precipitation={length_precip}"
            )
        return self


class WeatherResponse(BaseModel):
    """
    Top-level model for the weather API response.
    Contains geographic metadata and a nested hourly data block.
    """
    latitude: float = Field(ge=-90, le=90, description="Latitude of the location")
    longitude: float = Field(ge=-180, le=180, description="Longitude of the location")
    timezone: str
    hourly: HourlyBlock
