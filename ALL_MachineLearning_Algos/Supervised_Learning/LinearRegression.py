import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
# Printing the data set on terminal in table format
df['Target'] = data.target
print(df.head())
from sklearn.linear_model import LinearRegression

# Separate features and target
X = df[data.feature_names]
y = df['Target']

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Optionally, print R^2 score for training data
print("R^2 score (training):", model.score(X, y))
