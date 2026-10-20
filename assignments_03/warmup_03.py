import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing Question 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled.mean(axis=0))
# We fit scaler on the X_train set only to avoid leaking the test set, 
# ensuring that no information from the test set influences our preprocessing or model training.

# Preprocessing Question 2
encoder = OneHotEncoder(sparse_output=False)
seasons =  [["winter"], ["spring"], ["summer"], ["fall"]]
encoded = encoder.fit_transform(seasons)

print(encoded.shape)
print(encoder.get_feature_names_out())

# The single season column became 4 columns.
# This is because One-Hot Encoding creates exactly one new binary columns for
# every unique category found in the original column.

# Preprocessing Question 3
numeric_features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
categorical_features = ["season"]

X = df[numeric_features + categorical_features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy of the model: {accuracy_score(y_test, y_pred)}")
# When we call model.fit(X_train, y_train), the pipeline fits the scaler 
# and the encoder on the training data only, transforms it, 
# and fits the classifier on the result. When we call model.predict(X_test), 
# it applies the same scaling and encoding, learned from the training data, 
# to the test data before predicting.

# KNN Question 1
X = df[numeric_features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline(
    steps=[
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))

# KNN Question 2
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

# KNN Question 3
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
print("Unscaled accuracy:", accuracy_score(y_test, knn_unscaled.predict(X_test)))

knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_scaled.fit(X_train, y_train)
print("Scaled accuracy:  ", accuracy_score(y_test, knn_scaled.predict(X_test)))

# Scaling raised accuracy from about 0.925 to about 0.958. 
# That improvement comes almost entirely from letting smaller-range features count fairly in the distance. 
# For a distance-based model, scaling is not optional bookkeeping. Scaling allows to place all features on the 
# comparable footing and not making features with the larger range dominate in the model.

# Logistic Regression Question 1
X = df[numeric_features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

# Print the results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision ('good'): {precision:.4f}")
print(f"Recall ('good'): {recall:.4f}")
print(f"F1 Score ('good'): {f1:.4f}")

# Logistic Regression Question 2
logreg = model.named_steps["logreg"]
for name, coef in zip(numeric_features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")

# precipitation and wind speed have large negative coefficients, which means more rain and more wind 
# both push a day strongly toward "skip." it matches my intuition about the running weather.
# Rain and wind are the usual deal-breakers, and the model discovered that.

# Logistic Regression Question 3
probabilities = model.predict_proba(X_test[:5])[:, 1]   # probability of "good"
print(probabilities.round(2))

# A prediction of 0.96 is a confident "good," while 0.47 is less than 0.5 threshold based on that
# day will be marked as "skip" though it might not be that bad.

# Evaluation Question 1
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["skip", "good"])
disp.plot(colorbar=False)
plt.title("Test Confusion Matrix")
plt.show()
plt.savefig("outputs/logreg_confusion_matrix.png")

# Number of False Positives = 7
# Number of False Negatives = 9
# A false positive means that the model says a bad day is good, and the app sends you out into bad weather.
# For a running application this kind of error would matter more, which means that we need to make sure it is low.

# joblib Question 1
joblib.dump(model, "models/warmup_model.pkl")

loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == model.predict(X_test)).all()
print("Predictions match.")

# A trained logistic regression model knows its coefficients, but it does not know the scale of the data it was trained on. 
# If we save only the classifier and later call .predict() on raw, unscaled data, the predictions will be wrong, and
# our assertion will be failed at this case.