# Task 2: The Boundary -- weatherkit/schemas.py
import json

from weatherkit import WeatherResponse
from weatherkit import to_readings
from weatherkit import DailyAggregator


def main():
    with open("weather_raw.json", "r") as f:
            raw_data = json.load(f)
    
    
    # Validate the dictionary using the Pydantic model
    validated_weather = WeatherResponse.model_validate(raw_data)
    
    print(f"Latitude: {validated_weather.latitude}")
    print(f"Timezone: {validated_weather.timezone}")
    print(f"Hourly Observations: {len(validated_weather.hourly.time)}")

    # Convert the validated data into a list of HourlyReading objects
    hourly_readings = to_readings(validated_weather)

    # Aggregate to DailySummary objects
    daily_aggregator = DailyAggregator(min_hours=24)
    daily_summaries = daily_aggregator.summarize(hourly_readings)
    incomplete_days = daily_aggregator.incomplete_days(hourly_readings)


    print(f"{'date':<12} | {'temp_max':>6} | {'temp_min':>6} | {'precipitation_sum':>7} | {'hours_observed':>5}")
    for day in daily_summaries:
        print(f"{day.date:<12} | {day.temp_max:>3}° | {day.temp_min:>3}° | {day.precipitation_sum:>6.2f} | {day.hours_observed:>4}")

    print("\nWarning: The following incomplete days were dropped:", ", ".join(incomplete_days))



# This guard ensures that the code below only runs when this script is executed directly, but 
# not when it is imported as a module in another script.
if __name__ == "__main__":
    main()

# Your WeatherResponse rejects the whole file if a single temperature is null. 
# Is that the right behavior for a weather pipeline? Describe one situation where you would want it, 
# and one where you would rather tolerate the gap. What would you change in the schema to tolerate it?

# Whether it is right behavir or not depends on the environment where pipeline is executed.
# Strict validation is requried when pipeline feeds critical system where missing data could cause
# a catastrophic failure or imputing the missing value is too risky. 
# For example if this data feeds an aviation routing syste or a financial trading algorithm.
# it is acceptiable to tolerate the gap in general analytis, reporting, dashboarding use cases.

# DailyAggregator.min_hours defaults to 24. What goes wrong if a pipeline runs at noon and 
# the day is only half over? How does incomplete_days() help?

# if a pipeline runs at noon, the current day only has about 12 hourly readings. 
# Because min_hours demands 24 readings, the aggregator will reject the current day 
# and drop it from the final summaries list. incomplete_days() prevents silent failure by
# acting as an audit log.

# You wrote weatherkit as a package rather than one file. Name one concrete thing that becomes easier in Week 10, 
# when a pipeline needs to import this code.

# Structuring it as a package with an __init__.py create a clean public API.
# When pipeline needs to import the code, it can import directly from the top-level
# weatherkit package without needing to know the internal file structure.
# This means we can reorganize, rename or split internal files in Week 10
# without breaking the pipelines's import statements.