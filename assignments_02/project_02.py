import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

# Task 2: Look at the Features
print(df.describe())
# count value is identical across all columns and equal to 365 which is equal to the total number of rows in the dataset. 
# it confirms there are no missing values.

correlations = df.corr(numeric_only=True)['temperature_2m_max'].sort_values(ascending=False)
print(correlations)
# temperature_2m_min relates most strongly with the temperature_2m_max (correlation coefficient = 0.920607).

plt.figure(figsize=(8, 6))
plt.scatter(
    df['temperature_2m_min'],
    df['temperature_2m_max'],
    color='teal',
    alpha=0.7
)
plt.title("Daily High vs Low Temperature")
plt.xlabel("Minimum Temperature (°C)")
plt.ylabel("Maximum Temperature (°C)")

plt.savefig("outputs/low_vs_high_api_data.png")
# The scatter plot shows a strong positive linear relationship between 
# temperature_2m_min and temperature_2m_max. 
# This means that as the daily minimum temperature increases, 
# the daily maximum temperature also increases.

# Task 3: Engineer a Feature
df['is_summer'] = df['date'].dt.month.isin([6, 7, 8]).astype(int)
print(df['is_summer'].value_counts())

# is_summer is highly likely to help to predict the daily high temperature.
# Weather follows strong seasonal patterns, and summer months naturally have
# higher baseline temperatures than the rest of the year.
# Adding this feature gives the model information about the season, 
# which can help it adjust its predictions for daily high temperature.

# Task 4: Baseline Model
X = df[["temperature_2m_min"]]
y = df["temperature_2m_max"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Slope of the Baseline Model:", model.coef_[0])

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE of the Baseline Model:", rmse)
print("R² of the Baseline Model:", r2)
# RMSE is in degress Celsius. Predications are typically pff by a little more than 3 degrees.

# Task 5: Full Model
feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]

X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)
y_pred_test = model_full.predict(X_test)
y_pred_train = model_full.predict(X_train)

r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"R² of the Full Model on the train set:{r2_train}")
print(f"R² of the Full Model on the test set:{r2_test}")
print(f"RMSE of the Full Model on the test set: {rmse_test}")
# Comparing the models shows that adding more features noticeably improved performance.
# The test R² increased from roughly 0.86 to 0.90, meaning the new full model 
# can explain about 90% of the variance in the daily high temperature (a 4% improvement). 
# Furthermore, the RMSE dropped from 3.15 to 2.67, meaning our new model's predictions 
# are, on average, about half a degree closer to the actual daily high temperatures.

for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Interpretation of Coefficients
# Features that raise the predicted high: temperature_2m_min (+0.925) and is_summer (+0.956).
# Features that lower the predicted high: precipitation_sum (-0.140) and wind_speed_10m_max (-0.092).
# These signs make logical sense. Higher minimum temperatures 
# and summer months naturally lead to hotter days. Conversely, precipitation 
# and high winds tend to suppress daytime heating, lowering the max temperature.
# Train vs. Test R² Comparison 
# The Train R² (0.873) and Test R² (0.900) are very close,
# the test score is actually slightly higher than the train score, 
# it tells us that the model generalizes very well to unseen data and is not overfitting.

# Task 6: Evaluate and Summarize
fig, ax = plt.subplots()
plt.scatter(y_pred_test, y_test, color="darkblue", alpha=0.7, label="Data points")
ax.axline((0, 0), slope=1, color='red', linestyle='--', label="Perfect Model (y=x)")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted High Temperature")
plt.ylabel("Actual High Temperature")
plt.legend()

plt.show()
plt.savefig("outputs/predicted_vs_actual_high.png")
# Based on the Predicted vs Actual plot, the error is not perfectly even across the range.
# The model performs exceptionally well at the high end, with points tightly clustered 
# around the reference line. However, it struggles more in the middle 
# and lower-middle ranges, where the points spread out much 
# further vertically from the line, indicating larger prediction errors for milder days.

# Summary
# 
# 1. Dataset Size: 
# We analyzed a full year of weather data (365 days). We split this into an 
# 80% training set (292 days) to teach the model, and a 20% test set (73 days) 
# to evaluate its performance on unseen data.
# 
# 2. Model Performance (RMSE and R²): 
# Our full model achieved an R² of 0.90 and an RMSE of 2.67 on the test set. 
# In practical terms, this R² means our features successfully explain 90% of 
# the changes in the daily high temperature. The RMSE means that when the model 
# makes a prediction, its guess is typically off by about 2.7 degrees Celsius.
# 
# 3. Largest Effect on Predicted High: 
# While the daily minimum temperature sets the baseline (+0.925 degrees for every 
# 1-degree increase in the low), the 'is_summer' feature has the largest individual 
# coefficient (+0.956). This means that simply being in June, July, or August 
# pushes the predicted high temperature up by nearly a full degree, acting as 
# a strong positive multiplier. 
# 
# 4. Surprising Result: 
# One surprising result from our diagnostic plots is that the model is actually 
# most accurate at predicting extreme heat (days over 25°C), but struggles with 
# a much wider margin of error during milder, transitional days (10°C to 25°C). 
# Additionally, the relatively small negative impact of precipitation (-0.140) 
# is somewhat surprising, as rainy days often intuitively feel much colder to us 
# than a mere fraction of a degree.