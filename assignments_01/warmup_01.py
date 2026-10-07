import pytest

from dataclasses import dataclass, FrozenInstanceError, field
from pydantic import BaseModel, ValidationError, Field, model_validator


# --- Classes ---
# Q1

class Thermometer:
    def __init__(self, location: str, readings: list = None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading:float):
        self.readings.append(reading)

    def average(self):
        # we need to check if there are any readings to avoid division by zero
        if not self.readings:
            return 0.0
        
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        # we need to check if there are any readings to avoid division by zero
        if not self.readings:
            return None

        return max(self.readings)

    # Q2
    def __repr__(self):
        return f"location={self.location}, n_readings={len(self.readings)}, average={self.average():.2f}"


# Q3
class TemperatureAlert:
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer):
        return [reading for reading in thermometer.readings if reading > self.threshold]


# Dataclass Question 1
@dataclass(frozen=True)
class Station:
    """ Station represents a weather station with its unique identifier, name, geographical coordinates, and elevation."""
    station_id: str # Unique identifier for the station
    name: str # Name of the station
    latitude: float # Latitude of the station
    longitude: float # Longitude of the station
    elevation: float # Elevation of the station in meters


# Dataclass Question 3
@dataclass
class StationBatch:
    """ StationBatch represents a batch of weather stations. """
    region: str # Region name for the batch of stations
    # ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
    # python refuses the mutable default value for the field stations because it can lead to 
    # unexpected behavior if the same list is shared across multiple instances of StationBatch.
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """ Adds a new station to the batch. """
        self.stations.append(station)

    def highest(self) -> Station | None:
        """ Returns the station with the highest elevation in the batch. """
        if not self.stations:
            return None
        return max(self.stations, key=lambda s: s.elevation)


# Pydantic Question 1
class Reading(BaseModel):
    """ Reading represents a temperature, humidity reading for a particular station at a specific timestamp. """
    station_id: str = Field(min_length=3) # Unique identifier for the station
    timestamp: str # Timestamp of the reading in ISO 8601 format
    temperature_c: float = Field(ge=-90, le=60, description="Temperature in Celsius between -90 and 60")
    humidity: float = Field(ge=0, le=100, description="Humidity percentage between 0 and 100")

    @model_validator(mode="after")
    def check_sensor_health(self) -> 'Reading':
        """ Validates the sensor health based on temperature and humidity readings. """
        # This rule cannot be express with Field constraints, because it involves a relationship between two fields (temperature and humidity), 
        # while Field constraints only validate individual fields independently.
        if self.humidity == 0.0 and self.temperature_c < -40.0:
            raise ValueError("Invalid Sensor reading: humidity is 0% and temperature is below -40°C, indicating a potential sensor malfunction.")
        return self


# pytest Question 1
def celsius_to_fahrenheit(celsius: float) -> float:
    """ Converts Celsius to Fahrenheit. """
    return (celsius * 9 / 5) + 32


def test_celsius_to_fahrenheit():
    """ Tests the celsius_to_fahrenheit function. """
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6, 0.1)


# pytest Question 2
def mean(values: list[float]) -> float:
    """ Calculates the mean of a list of values. """
    if not values:
        raise ValueError("The list of values is empty, cannot compute mean.")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    """ Test that mean raises ValueError when given an empty list. """
    # match argument checks the error message against a regular expression, without match, pytest.raises(ValueError)
    # passes if any ValueError occurs, including one caused by a typo in the setup.
    with pytest.raises(ValueError, match="The list of values is empty, cannot compute mean."):
        mean([])


# pytest Question 3
# Using parametrized test with four cases is better than writing four separate test functions
# because it reduced code duplication, makes the tests easier to read and maitain, 
# and allows for easier addition of new test cases in the future.
# ================================================= 7 passed in 0.14s =================================================
@pytest.mark.parametrize(
    "values, expected_mean",
    [
        ([1.0, 2.0, 3.0], 2.0),             # Standard list of positive numbers
        ([42.0], 42.0),                     # Single-element list
        ([10.0, -20.0, 4.0], -2.0),         # List containing negative numbers
        ([0.0, 0.0, 0.0], 0.0),             # List of zeros
        ([1.5, 2.5, 3.5, 4.5], 3.0),        # Fractional floats
    ]
)
def test_mean_values(values, expected_mean):
    assert mean(values) == expected_mean


# pytest Question 4
# ===================================================== FAILURES ======================================================
# ____________________________________________ test_celsius_to_fahrenheit _____________________________________________

#     def test_celsius_to_fahrenheit():
#         """ Tests the celsius_to_fahrenheit function. """
#         assert celsius_to_fahrenheit(0) == 32
# >       assert celsius_to_fahrenheit(100) == 212
# E       assert 257.0 == 212
# E        +  where 257.0 = celsius_to_fahrenheit(100)

# warmup_01.py:105: AssertionError
# ============================================== short test summary info ==============================================
# FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
# In the failure report above pytest shows the test function that failed, the line number where the failure occurred, 
# and the specific assertion that failed. it is more useful than bare "assertion failed" because it provides 
# neccessary context to understand what went wrong and where to look in the code to understand the issue.


if __name__ == "__main__":
    # Q1
    thermometer = Thermometer("Apex")
    thermometer.add(22.5)
    thermometer.add(23.0)
    thermometer.add(21.8)
    thermometer.add(24.1)

    print(f"Average temperature in {thermometer.location}: {thermometer.average():.2f}°C")
    print(f"Hottest temperature in {thermometer.location}: {thermometer.hottest():.2f}°C")

    # Q2
    # if there is no represenation for the object then a memory address is returned which is not
    # very useful.
    print(thermometer)
    observations = [
        Thermometer("Apex", [22.5, 23.0, 21.8, 24.1]),
        Thermometer("Boulder", [19.5, 20.0, 18.8, 21.1])
    ]
    print(observations)

    # Q3
    alert1 = TemperatureAlert(23.0)
    alert2 = TemperatureAlert(25.0)

    print(f"Readings above {alert1.threshold}°C:{alert1.breaches(thermometer)}")
    print(f"Readings above {alert2.threshold}°C:{alert2.breaches(thermometer)}")

    # why is the threshold stored on TemperatureAlert rather than passed as an argument to breaches()? 
    # What advantage does that give you if you have twenty thermometers to check?
    # Threshold is stored on TemperatureAlert so that it can be reused for multiple thermometers 
    # without having to specify it each time.

    # Dataclass Question 1
    station_a = Station("001", "Station A", 40.7128, -74.0060, 10.0)
    station_b = Station("002", "Station B", 34.0522, -118.2437, 15.0)

    # Check if two stations are equal based on their attributes. @dataclass automatically generates 
    # the __eq__ method for us, which compares the attributes of the instances.
    print(station_a == station_b)  # False

    
    # Dataclass Question 2
    try:
        station_a.latitude = 41.0  # This will raise an error because the dataclass is frozen (immutable)
    except FrozenInstanceError as e:
        print(f"Error: {e}")

    # frozen = True means that the attributes of the dataclass cannot be modified after the instance is created.
    # But also it means that the instances of dataclass are hashable and can be used in sets or 
    # as dictionary keys. This is useful for ensuring that the data remains consistent and unchanged.
    stations = set([station_a, station_a, station_b])
    print(f"Number of unique stations: {len(stations)}")

    # Pydantic Question 1
    station_read = Reading(station_id="001", timestamp="2023-10-01T12:00:00Z", temperature_c=25.0, humidity=50.0)
    print(station_read)

    # Pydantic Question 2
    try:
        missing_field = Reading(station_id="002", timestamp="2023-10-01T12:00:00Z", temperature_c=25.0)
    except ValidationError as e:
        print(f"Error: {e}")

    try:
        invalid_temp = Reading(station_id="003", timestamp="2023-10-01T12:00:00Z", temperature_c=150.0, humidity=50.0)
    except ValidationError as e:
        print(f"Error: {e}")

    try:
        invalid_humidity = Reading(station_id="004", timestamp="2023-10-01T12:00:00Z", temperature_c=25.0, humidity="very humid")
    except ValidationError as e:
        print(f"Error: {e}")

    reading = Reading(station_id="005", timestamp="2023-10-01T12:00:00Z", temperature_c="21.5", humidity=40)
    print(reading)
    print(f"Temperature type: {type(reading.temperature_c)}, Humidity type: {type(reading.humidity)}")
    # pydantic automatically converts compatible types, so the string "21.5" is converted to a float and the integer 40 is converted to a float as well.
    # while "very humid" is not a compatible type for humidity, so it raises a validation error.

    # Pydantic Question 3
    try:
        invalid_reading = Reading(
            station_id="01", 
            temperature_c="high", 
            humidity=50.0
        )
    except ValidationError as e:
        # 3 errors will be reported, reporting all of them at once is more efficient than reporting them one by one
        # because it allows the user to see all the issues with the input data at once, 
        # rather than having to fix one issue and then run the validation again to find the next issue.
        for error in e.errors():
            print(f"Error in field '{error['loc'][0]}': {error['msg']}")

    # Pydantic Question 4
    # 1. Demonstrating a valid reading
    try:
        valid_reading = Reading(station_id="006", timestamp="2023-10-01T12:00:00Z", temperature_c=20.0, humidity=50.0)
        print(f"Success: {valid_reading}")
    except ValidationError as e:
        print(e)

    # 2. Demonstrating a valid reading with extreme cold but normal humidity
    try:
        cold_reading = Reading(station_id="007", timestamp="2023-10-01T12:00:00Z", temperature_c=-50.0, humidity=20.0)
        print(f"Success: {cold_reading}")
    except ValidationError as e:
        print(e)

    # 3. Demonstrating the rejected combination
    try:
        bad_reading = Reading(station_id="008", timestamp="2023-10-01T12:00:00Z", temperature_c=-45.0, humidity=0.0)
        print("Success:", bad_reading)
    except ValidationError as e:
        print(f"\nFailed as expected:\n{e}")
