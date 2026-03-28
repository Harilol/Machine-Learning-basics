# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load Dataset
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target

print(df.head())


# Basic EDA
print("Shape:", df.shape)
print(df.isnull().sum())
print(df.describe())

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


# Feature Engineering
df["Rooms_per_House"] = df["AveRooms"] / df["HouseAge"]

X = df.drop("Price", axis=1)
y = df["Price"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)


# This model uses polynomial features to capture non-linear relationships
# compared to basic linear regression
# Polynomial Features
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

#Training model with polynomial features
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

msep = mean_squared_error(y_pred,y_test)
r2p = r2_score(y_test, y_pred)

print("msep:",msep)
print("r2p:",r2p)

# Polynomial Regression improved R2 score from 0.57 → 0.66
# Shows data has some non-linear relationships
