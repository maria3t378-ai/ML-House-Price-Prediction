
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv("data/house_prices.csv")


# Features and target
X = df.drop("price", axis=1)
y = df["price"]


# Categorical and numerical columns
categorical_features = ["location"]

numerical_features = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Machine Learning model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
pipeline.fit(X_train, y_train)


# Predictions
y_pred = pipeline.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Display results
print("Model training completed!")
print(f"Mean Absolute Error: {mae:,.2f}")
print(f"R² Score: {r2:.4f}")


# Save trained model
joblib.dump(
    pipeline,
    "models/house_price_model.pkl"
)

print("Model saved successfully!")
print("Saved at: models/house_price_model.pkl")
