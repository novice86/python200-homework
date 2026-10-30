# tests/test_classifier.py
import pytest
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from weather_model import WeatherClassifier, Prediction


@pytest.fixture
def model_path(tmp_path):
    """Train a tiny model on obvious days and save it to a temp file."""
    # 15 perfect days
    good = [{"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0.0, "wind_speed_10m_max": 10}] * 15
    
    # A mix of totally bad days AND single-factor bad days
    bad_scenarios = [
        {"temperature_2m_max": 2, "temperature_2m_min": -6, "precipitation_sum": 12.0, "wind_speed_10m_max": 40},  # All bad
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 12.0, "wind_speed_10m_max": 15}, # Rainy only
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 45},  # Windy only
        {"temperature_2m_max": 3, "temperature_2m_min": -2, "precipitation_sum": 0.0, "wind_speed_10m_max": 15}    # Cold only
    ]
    # Multiply by 4 so we have 16 bad days to balance the 15 good days
    bad = bad_scenarios * 4 
    
    X = pd.DataFrame(good + bad)[WeatherClassifier.FEATURES]
    y = [1] * 15 + [0] * 16

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)

    path = tmp_path / "weather_classifier.pkl"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def clf(model_path):
    """A WeatherClassifier loaded from the saved tiny model."""
    return WeatherClassifier(model_path)


def test_good_day_predicted_good(clf):
    result = clf.predict([
        {"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0.0, "wind_speed_10m_max": 10}
    ])
    assert result[0].label == "good"


@pytest.mark.parametrize("input_features, expected_label", [
    # 1. Windy only: Perfect temperatures and dry, but wind is 45 km/h (well over 30)
    ({"temperature_2m_max": 18.0, "temperature_2m_min": 10.0, "precipitation_sum": 0.0, "wind_speed_10m_max": 45.0}, "skip"),

    # 2. Rainy only: Perfect temperatures and calm wind, but 12mm of heavy rain
    ({"temperature_2m_max": 18.0, "temperature_2m_min": 10.0, "precipitation_sum": 12.0, "wind_speed_10m_max": 15.0}, "skip"),

    # 3. Cold only: Dry and calm wind, but freezing temperatures (high 3°C, low -2°C)
    ({"temperature_2m_max": 3.0, "temperature_2m_min": -2.0, "precipitation_sum": 0.0, "wind_speed_10m_max": 15.0}, "skip"),

    # 4. All of it: cold, heavy rain, and wind
    ({"temperature_2m_max": 3.0, "temperature_2m_min": -2.0, "precipitation_sum": 12.0, "wind_speed_10m_max": 45.0}, "skip")
])
def test_bad_day_predicted_skip(clf, input_features, expected_label):
    result = clf.predict([input_features])
    assert result[0].label == expected_label


def test_probability_is_between_zero_and_one(clf):
    result = clf.predict([
        {"temperature_2m_max": 15, "temperature_2m_min": 8, "precipitation_sum": 1.0, "wind_speed_10m_max": 12}
    ])
    assert 0.0 <= result[0].probability <= 1.0


def test_one_prediction_per_day_in_order(clf):
    days = [
        {"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0.0, "wind_speed_10m_max": 10},
        {"temperature_2m_max": 2, "temperature_2m_min": -6, "precipitation_sum": 12.0, "wind_speed_10m_max": 40},
        {"temperature_2m_max": 10, "temperature_2m_min": 5, "precipitation_sum": 5.0, "wind_speed_10m_max": 20},
    ]
    results = clf.predict(days)
    assert len(results) == 3
    assert all(isinstance(r, Prediction) for r in results)


def test_empty_list_returns_empty(clf):
    assert clf.predict([]) == []


def test_missing_feature_raises(clf):
    with pytest.raises(ValueError, match="missing required features"):
        clf.predict([{"temperature_2m_max": 18, "temperature_2m_min": 9}])   # no precipitation_sum or wind_speed_10m_max


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file"):
        WeatherClassifier(tmp_path / "does_not_exist.pkl")

# I modified classifier.py to return None when the input days list is empty. 
# As expected, test_empty_list_returns_empty successfully caught this regression and failed.
# E       assert None == []
# E        +  where None = predict([])
# E        +    where predict = <weather_model.classifier.WeatherClassifier object at 0x118efb790>.predict
#
# tests/test_classifier.py:92: AssertionError
# ========================================================== short test summary info ===========================================================
# FAILED tests/test_classifier.py::test_empty_list_returns_empty - assert None == []
# ======================================================== 1 failed, 9 passed in 2.79s ============================

