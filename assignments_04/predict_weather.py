# Task 5: The Script
from weather_model import WeatherClassifier

SAMPLE_DAYS = [
    {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},
    {"temperature_2m_max": 22, "temperature_2m_min": 13, "precipitation_sum": 9.0, "wind_speed_10m_max": 14},
    {"temperature_2m_max": 15, "temperature_2m_min": 5, "precipitation_sum": 2.9, "wind_speed_10m_max": 10},
    {"temperature_2m_max": 27, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 15},
]


def main() -> None:
    """Load the component and print a prediction for each sample day."""
    classifier = WeatherClassifier("models/weather_classifier.pkl")
    for day, result in zip(SAMPLE_DAYS, classifier.predict(SAMPLE_DAYS)):
        print(f"{day['temperature_2m_max']:.0f}C, {day['precipitation_sum']:.0f}mm rain, "
              f"{day['wind_speed_10m_max']:.0f} km/h wind -> {result.label} ({result.probability:.2f})")


# This guard prevents the main prediction logic from running automatically 
# if another script imports predict_weather.py just to use its helper functions.
if __name__ == "__main__":
    main()

# Task 6: Reflection
# 1. Enforcing a fixed feature order prevents silent failure of the classifier  due to misaligned columns.
# 2. Training tiny own model is better design for the test suite than depending on the real life file because
#    it satisfy the main purpose of the tests - check the structural integrity of the code of the WeatherClassifier
#    class - ensuring it correctly formats inputs and structures the outputs for the downstream pipeline. 
#    Training a tiny, lightweight model in a fixture save execution time and avoide the overhead of loading a heavy production file.
# 3. Why it is ready: predict() method is already designed to process a list of days (batch processing) 
#    What to add: Before moving to production, we should add strict data validation at the input boundary to ensure 
#    the classifier does not crash on invalid or missing data when making predictions on database records.