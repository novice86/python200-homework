from dataclasses import dataclass
from weatherkit import HourlyReading
from collections import defaultdict

@dataclass
class DailySummary:
    """
    Represents a single day's weather summary.
    
    Attributes:
        date (str): The date of the summary.
        temp_max (float): The maximum air temperature for the day in degrees Celsius.
        temp_min (float): The minimum air temperature for the day in degrees Celsius.
        precipitation_sum (float): The total precipitation for the day in millimeters.
        hours_observed (int): The number of hours for which observations were recorded.
    """
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """
        Calculates the temperature range for the day.

        Returns:
            float: The difference between the maximum and minimum temperatures.
        """
        return self.temp_max - self.temp_min


class DailyAggregator:
    """
    Aggregates hourly weather readings into daily summaries.
    """

    def __init__(self, min_hours = 24):
        """
        Initializes the DailyAggregator with a minimum number of hours required for a valid summary.

        Args:
            min_hours (int): The minimum number of hourly readings required to create a daily summary.
        """
        self.min_hours = min_hours

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """
        Aggregates a list of HourlyReading objects into daily summaries.

        Args:
            readings (list[HourlyReading]): A list of hourly weather readings.

        Returns:
            list[DailySummary]: A list of daily summaries, one for each unique date in the readings.
        """

        daily_data = defaultdict(list)

        # Group readings by date
        for reading in readings:
            date = reading.timestamp.split("T")[0]  # Extract the date part from the timestamp
            daily_data[date].append(reading)

        summaries = []

        # Create a DailySummary for each date
        for date, daily_readings in daily_data.items():
            if len(daily_readings) >= self.min_hours:
                temp_max = max(r.temperature_c for r in daily_readings)
                temp_min = min(r.temperature_c for r in daily_readings)
                precipitation_sum = sum(r.precipitation_mm for r in daily_readings)
                hours_observed = len(daily_readings)

                summaries.append(
                    DailySummary(
                        date=date,
                        temp_max=temp_max,
                        temp_min=temp_min,
                        precipitation_sum=precipitation_sum,
                        hours_observed=hours_observed
                    )
                )

        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """
        Identifies dates with fewer than the minimum required hourly readings.

        Args:
            readings (list[HourlyReading]): A list of hourly weather readings.

        Returns:
            list[str]: A list of dates that have fewer than the minimum required hourly readings.
        """

        daily_data = defaultdict(list)

        # Group readings by date
        for reading in readings:
            date = reading.timestamp.split("T")[0]  # Extract the date part from the timestamp
            daily_data[date].append(reading)

        incomplete_dates = [
            date for date, daily_readings in daily_data.items()
            if len(daily_readings) < self.min_hours
        ]

        return incomplete_dates