# 🏠 House Price Prediction System

A Machine Learning-based web application that predicts house prices based on property features such as area, bedrooms, bathrooms, stories, parking spaces, and location.

## 🚀 Project Overview

This project uses a **Random Forest Regression** model to estimate house prices.

The trained Machine Learning model is integrated with **Streamlit** to provide an interactive web application where users can enter house details and receive an estimated price.

## ✨ Features

* 🏠 House price prediction
* 📊 Interactive Streamlit web application
* 🤖 Random Forest Machine Learning model
* 📍 Location-based prediction
* 📈 Dynamic prediction visualizations
* 💰 Estimated house price in Pakistani Rupees
* 📊 House price distribution visualization
* 🔄 Real-time prediction based on user input

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

## 📂 Project Structure

```text
ML-House-Price-Prediction/
│
├── data/
│   └── house_prices.csv
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│
├── src/
│   ├── create_dataset.py
│   ├── train_model.py
│   ├── predict.py
│   └── visualize.py
│
├── app.py
├── requirements.txt
└── README.md
```

## 🤖 Machine Learning Model

The project uses:

**Algorithm:** Random Forest Regressor

**R² Score:** 0.9846

**Training Records:** 200

The model uses the following features:

* Area
* Bedrooms
* Bathrooms
* Stories
* Parking
* Location

## 📊 Visualizations

The Streamlit application dynamically generates visualizations based on the current prediction.

### Current House Predicted Price

Displays the predicted price for the house entered by the user.

### House Price Distribution

Shows the distribution of house prices in the dataset and highlights the current prediction.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/maria3t378-ai/ML-House-Price-Prediction.git
```

### 2. Open the project folder

```bash
cd ML-House-Price-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🎯 Project Goal

The goal of this project is to demonstrate an end-to-end Machine Learning workflow:

**Data → Preprocessing → Model Training → Evaluation → Prediction → Web Application**

## 👩‍💻 Author

**Maria Siddique**

AI & Machine Learning Developer

GitHub: `maria3t378-ai`
