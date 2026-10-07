import pandas as pd
import numpy as np

np.random.seed(42)

locations = ["Lahore", "Islamabad", "Karachi", "Rawalpindi", "Faisalabad"]

data = {
    "area": np.random.randint(600, 3500, 200),
    "bedrooms": np.random.randint(1, 6, 200),
    "bathrooms": np.random.randint(1, 5, 200),
    "stories": np.random.randint(1, 4, 200),
    "parking": np.random.randint(0, 4, 200),
    "location": np.random.choice(locations, 200)
}

df = pd.DataFrame(data)

location_price = {
    "Lahore": 8000,
    "Islamabad": 12000,
    "Karachi": 7500,
    "Rawalpindi": 6500,
    "Faisalabad": 5000
}

df["price_per_sqft"] = df["location"].map(location_price)

df["price"] = (
    df["area"] * df["price_per_sqft"]
    + df["bedrooms"] * 500000
    + df["bathrooms"] * 300000
    + df["stories"] * 200000
    + df["parking"] * 250000
)

df["price"] = df["price"] + np.random.randint(-500000, 500001, 200)

df.drop(columns=["price_per_sqft"], inplace=True)

df.to_csv("data/house_prices.csv", index=False)

print("Dataset created successfully!")
print(f"Total houses: {len(df)}")
print(df.head())