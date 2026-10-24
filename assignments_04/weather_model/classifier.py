from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import joblib


# Task 1: The Prediction Dataclass
@dataclass
class Prediction:
    """The model's verdict for a single day.

    Atrributes:
        label: Either "good" or "skip
        probability: The probability, from 0 to 1, that the day is good for running.
    """
    label: str
    probability: float


# Task 2: The WeatherClassifier Class

class WeatherClassifier:
    """Loads a saved weather pipeline one and makes prediction on many days"""
    FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ]
    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model file at {model_path}. Train and save the model first."
            )
        self.model = joblib.load(model_path)

    def predict(self, days: list[dict]) -> list[Prediction]:
        """Predict weather each day is good for running.

        Args:
            days: A list of dictionaries, each with the keys temperature_2m_max, temperature_2m_min,
                precipitation_sum, and wind_speed_10m_max.

        Returns:
            List of predictions: One prediction per input day, in the same order.

        """
        if not days:
            return []

        frame = pd.DataFrame(days)
        missing = [f for f in self.FEATURES if f not in frame.columns]
        if missing:
            raise ValueError(f"missing required features: {missing}")

        X = frame[self.FEATURES]
        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return [
            Prediction("good" if int(label) == 1 else "skip", float(prob))
            for label, prob in zip(labels, probabilities)
        ]


    

