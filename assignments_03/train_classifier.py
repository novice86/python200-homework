import requests
import pandas as pd
import numpy as np
import json
import sys
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

# Task 1: Fetch the Data

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.7272,
    "longitude": -78.8541,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print(f"Shape of the dataset: {df.shape}")
# print first five rows of the dataset collected for the city of Apex
print(df.head())


# A day is good when the high is between 7 and 26 degrees, the low is at least 0, precipitation is under 3 mm, and the maximum wind is under 30 km/h. 
df['good_for_running'] = np.where(
    (df['temperature_2m_max'] >= 7) & 
    (df['temperature_2m_max'] <= 26) & 
    (df['temperature_2m_min'] >= 0) & 
    (df['precipitation_sum'] < 3) & 
    (df['wind_speed_10m_max'] < 30),
    1, # If condition is met
    0  # If condition is not met
)

print("Class balance:\n", df['good_for_running'].value_counts())
# number of good days is 156 
# number of skip days is 209
# 43% of days are good, it seems reasonable for the climate of the Apex city.

# Task 3: Build and Evaluate a KNN Classifier
numeric_features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric_features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for k in [1, 3, 5, 7, 9, 11, 15, 19, 25]:
    knn = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ]
    )
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}")

# The best cross-validation score is around k = 11. The very small values do slightly worse, 
# and the largest values start to fall off as the model underfits.

best_k = 11
final_knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])
final_knn_model.fit(X_train, y_train)
y_pred = final_knn_model.predict(X_test)
print("Test accuracy of the knn model:", accuracy_score(y_test, y_pred))
print("Full classification report:\n", classification_report(y_test, y_pred))

# Task 4: Build and Evaluate a Logistic Regression Classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

logreg = pipeline.named_steps["logreg"]
coefficients = logreg.coef_[0]

print("\nFeature Coefficients:")
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef:.4f}")

# The condition that most strongly pushes a day toward "skip" (class 0) is precipitation_sum, 
# which has the largest negative coefficient (-3.0184). 
# The second strongest factor is temperature_2m_max (-0.9902), meaning higher maximum 
# temperatures also strongly increase the likelihood of the day being marked as a "skip".

# Task 5: Compare and Choose
# Accuracy of the linear regression model = 0.85 is higher than accuracy of the knn model.
# For a running application Precision is the most critical metric because of the high cost 
# of the False Positives. The user goes outside during bad weather conditions.
# Logistic regression is clear choise for deployment for the following reasons:
#  1. Its saved file is tiny, because it stores a handful of coefficients rather than 
#     the entire training set. KNN would have to carry all of its training data into the deployment 
#     and search through it for every prediction.
#  2. Its coefficients are interpretable, so its decisions can be explained.
#  3. It produces smooth probability confidence score, having these exact percentages allows us to 
#     adjust the threshold for a "good" run later if we want to be more or less strict.

# Task 6: Save the Model
model_filepath = "models/weather_classifier.pkl"
joblib.dump(pipeline, model_filepath)

metadata = {
    "python_version": sys.version,
    "scikit_learn_version": sklearn.__version__,
    "feature_names": list(X_train.columns),
    "label_rule": "high >= 7 and high <= 26, low >= 0, precipitation_sum < 3, wind_speed_10m_max < 30",
    "city": "Apex, North Carolina",
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_f1": float(f1_score(y_test, y_pred))
}
metadata_filepath = "models/weather_classifier_metadata.json"
with open(metadata_filepath, "w") as json_file:
    json.dump(metadata, json_file, indent=4)

print(f"Success! Model pipeline saved to {model_filepath}")
print(f"Metadata securely saved to {metadata_filepath}")