# scikit-learn Question 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)

new_lows = np.array([[6], [18]])
print(new_lows)
predictions = model.predict(new_lows)
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Predicted high for a low of 6: {predictions[0]:.2f}")
print(f"Predicted high for a low of 18: {predictions[1]:.2f}")

# scikit-learn Question 2

x = np.array([10, 20, 30, 40, 50])
print(f"Shape of 1D array{x.shape}")

x_2d = x.reshape(-1, 1)
print(f"New shape{x_2d.shape}")
# scikit-learn needs X to be 2D array because it assumes that each row represents a training example, each column represents a feature.

# Linear Regression
rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Linear Regression Question 1
plt.scatter(x=temp_min, y=temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Minimum Temperature")
plt.ylabel("Maximum Temperature")

plt.savefig("outputs/high_vs_low.png")

# We can see two visible groups on the plot that follow a linear pattern. 
# The location of the colored points—notably the tendency of the blue points 
# to sit slightly higher on the y-axis than the red points—suggests that 
# non-rainy days might reach slightly higher maximum temperatures than rainy days.

# Linear Regression Question 2
X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Linear Regression Question 3
model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope:4f}")
print(f"Intercept: {intercept:4f}")

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:4f}")
print(f"R^2: {r2:4f}")

# For every 1-degree increase in the daily low temperature (temp_min)
# the daily high temperature (temp_max) is expected to increase by the exact value of the slope, 
# in this case it is 0.94

# Linear Regression Question 4
X_full = np.column_stack([temp_min, is_rainy])

X_full_train, X_full_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
model_full = LinearRegression()
model_full.fit(X_full_train, y_train)

y_pred = model_full.predict(X_full_test)

rmse_full = np.sqrt(mean_squared_error(y_test, y_pred))
r2_full = r2_score(y_test, y_pred)

print(f"R^2 full: {r2_full:4f}")
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Based on the R^2 value adding the is_rainy flag helps better explain the variability in the daily high temperatureimproving the score 
# from 0.89 to 0.95.
# In practical terms, this coefficient means that if it is raining, 
# the expected daily high temperature decreases by 2.9 degrees compared 
# to a day with no rain (assuming the daily low temperature is the same).

# Linear Regression Question 5
fig, ax = plt.subplots()
plt.scatter(y_pred, y_test, color="darkblue", alpha=0.7, label="Data points")
ax.axline((0, 0), slope=1, color='red', linestyle='--', label="Perfect Model (y=x)")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted High Temperature")
plt.ylabel("Actual High Temperature")
plt.legend()

plt.savefig("outputs/predicted_vs_actual_warmup.png")

# A point falling ABOVE the diagonal line (y > x) means the true (actual) value is larger than the predicted value. 
# In this case, the model UNDERESTIMATED what the high temperature would be.
# 
# A point falling BELOW the diagonal line (y < x) means the true (actual) value is smaller than the predicted value. 
# In this case, the model OVERESTIMATED what the high temperature would be.