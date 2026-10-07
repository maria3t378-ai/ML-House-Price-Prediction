
import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/house_price_model.pkl")


print("\n🏠 House Price Prediction System")
print("--------------------------------")


# Get user input
area = float(input("Enter house area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
stories = int(input("Enter number of stories: "))
parking = int(input("Enter number of parking spaces: "))

print("\nAvailable locations:")
print("Lahore")
print("Islamabad")
print("Karachi")
print("Rawalpindi")
print("Faisalabad")

location = input("Enter location: ").strip()


# Create input data
house_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "parking": [parking],
    "location": [location]
})


# Make prediction
predicted_price = model.predict(house_data)[0]


# Display result
print("\n--------------------------------")
print(f"Estimated House Price: Rs. {predicted_price:,.0f}")
print("--------------------------------")

