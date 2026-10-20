# Task 7: Load and Predict

import joblib
import json
import pandas as pd

pipeline = joblib.load("models/weather_classifier.pkl")

with open("models/weather_classifier_metadata.json", "r") as file:
    metadata = json.load(file)

print("MODEL METADATA")
print(f"City: {metadata['city']}")
print(f"Features: {', '.join(metadata['feature_names'])}")
print(f"Test Accuracy: {metadata['test_accuracy']:.4f}\n")

# Build a DataFrame of hypothetical days
# We use metadata['feature_names'] to ensure the column order perfectly matches the training data
features = metadata['feature_names']

# Scenarios:
# 1. Clearly good (perfect temps, no rain, light wind)
# 2. Clearly bad (freezing cold, heavy rain, high wind)
# 3. Borderline precip (perfect temps, but 2.9mm rain is just under the 3.0mm cutoff)
# 4. Borderline temp (dry and calm, but 27°C is just barely over the 26°C max)
hypothetical_data = [
    [18.0, 10.0, 0.0, 15.0],  # Day 1: Good
    [2.0, -5.0, 15.0, 45.0],  # Day 2: Bad
    [15.0, 5.0, 2.9, 10.0],   # Day 3: Borderline 1
    [27.0, 15.0, 0.0, 15.0]   # Day 4: Borderline 2
]

df_new = pd.DataFrame(hypothetical_data, columns=features)

# Generate predictions and probabilities
predictions = pipeline.predict(df_new)
probabilities = pipeline.predict_proba(df_new)[:, 1] # Index 1 is the probability of "good"

# Map the numeric predictions (0 and 1) back to human-readable labels
label_map = {0: "skip", 1: "good"}

for i in range(len(df_new)):
    inputs = df_new.iloc[i].to_dict()
    pred = label_map[predictions[i]]
    prob = probabilities[i]
    
    # Format inputs for cleaner printing
    input_str = ", ".join([f"{k}: {v}" for k, v in inputs.items()])
    
    print(f"Day {i+1} | {input_str}")
    print(f"--> PREDICTION: {pred.upper()} (Confidence: {prob:.1%} chance of being good)\n")

# Logistic Regression model labeled Day 3 as "SKIP" (39.4%). Because the model applies a 
# massive negative weight (-3.0184) to precipitation. The model learned that 2.9mm is practically the same as 3.0mm,
# smoothing out the rigid cutoff. Similarly on Day 4, pushing the max temperature just one degree over 
# comfortable 26°C limit (to 27°C) caused the probability to drop to 40.1%, correctly triggering a "SKIP".
# A probability of 0.52 indicates extreme uncertainty—the model is essentially tossing a coin. Because we established 
# that False Positives (sending a runner out into bad weather) are the worst  possible user experience, 
# I would handle a 0.52 prediction by leveling up the decision threshold.

# If predict.py is run before train_classifier.py, the script will crash with a  FileNotFoundError because 
# "models/weather_classifier.pkl" and the metadata  JSON file do not exist yet. 
# To make the error message more helpful, we could  wrap the file loading steps in a `try...except FileNotFoundError:` block. 
# Instead of a massive Python traceback, the except block would print a clean  message:
# "Error: Model files not found. Please run train_classifier.py first."

# To run automatically in a cloud pipeline, predict.py cannot rely on hardcoded  hypothetical data or 
# terminal print statements. It would need to be refactored 
# to dynamically accept live data  Additionally, instead  of printing to the console, 
# the script must export the final predictions downstream,  such as writing the "good/skip" labels directly 
# into a database table or returning a JSON response for another microservice to consume.