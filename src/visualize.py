
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("data/house_prices.csv")


# Create output folder
import os
os.makedirs("visualizations", exist_ok=True)


# ==========================================================
# Graph 1: House Price Distribution
# ==========================================================

plt.figure(figsize=(10, 6))

plt.hist(df["price"], bins=20)

plt.title("House Price Distribution")
plt.xlabel("House Price (Rs.)")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.savefig("visualizations/price_distribution.png")
plt.close()


# ==========================================================
# Prepare data for prediction comparison
# ==========================================================

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Load trained model
model = joblib.load("models/house_price_model.pkl")


# Make predictions
y_pred = model.predict(X_test)


# ==========================================================
# Graph 2: Actual vs Predicted Prices
# ==========================================================

plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price (Rs.)")
plt.ylabel("Predicted House Price (Rs.)")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction reference line
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.tight_layout()
plt.savefig("visualizations/actual_vs_predicted.png")
plt.close()


print("Visualization completed successfully!")
print("Created:")
print("1. visualizations/price_distribution.png")
print("2. visualizations/actual_vs_predicted.png")

