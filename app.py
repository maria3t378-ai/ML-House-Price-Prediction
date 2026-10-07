import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# -----------------------------
# Load Model & Dataset
# -----------------------------
model = joblib.load("models/house_price_model.pkl")
df = pd.read_csv("data/house_prices.csv")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 20px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

.price {
    font-size: 38px;
    font-weight: 700;
}

.footer {
    text-align: center;
    margin-top: 50px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">🏠 House Price Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered house price estimation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Prediction Section
# -----------------------------
left_column, right_column = st.columns(2)


# -----------------------------
# House Details
# -----------------------------
with left_column:

    st.markdown(
        '<div class="section-title">🏡 House Details</div>',
        unsafe_allow_html=True
    )

    st.write("Enter the property information below.")

    area = st.number_input(
        "House Area (sq ft)",
        min_value=300,
        max_value=10000,
        value=2000,
        step=100
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    stories = st.number_input(
        "Stories",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

    location = st.selectbox(
        "Location",
        [
            "Lahore",
            "Islamabad",
            "Karachi",
            "Rawalpindi",
            "Faisalabad"
        ]
    )

    predict_button = st.button(
        "🔮 Predict House Price",
        use_container_width=True
    )


# -----------------------------
# Prediction Result
# -----------------------------
with right_column:

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if predict_button:

        # Create input data
        house_data = pd.DataFrame({
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],
            "parking": [parking],
            "location": [location]
        })

        # Machine Learning Prediction
        predicted_price = model.predict(house_data)[0]

        # Save prediction in session state
        st.session_state["predicted_price"] = predicted_price

        st.success("Prediction completed successfully!")

        st.markdown(
            f"""
            <div class="result-box">
                <div>💰 Estimated House Price</div>
                <div class="price">
                    Rs. {predicted_price:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "Prediction generated using a Random Forest "
            "Machine Learning model."
        )

    else:

        st.info(
            "👈 Enter the house details and click "
            "**Predict House Price**."
        )


# -----------------------------
# Model Information
# -----------------------------
st.divider()

st.markdown(
    '<div class="section-title">🤖 Model Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Algorithm",
        "Random Forest"
    )

with col2:
    st.metric(
        "R² Score",
        "0.9846"
    )

with col3:
    st.metric(
        "Training Records",
        "200"
    )


# -----------------------------
# Dynamic Prediction Graphs
# -----------------------------
st.divider()

st.markdown(
    '<div class="section-title">📈 Prediction Visualizations</div>',
    unsafe_allow_html=True
)


# Check if prediction exists
if "predicted_price" in st.session_state:

    predicted_price = st.session_state["predicted_price"]


    # =====================================================
    # GRAPH 1 — CURRENT PREDICTED PRICE
    # =====================================================

    st.subheader("🎯 Current House Predicted Price")

    fig1, ax1 = plt.subplots(figsize=(9, 5))

    ax1.bar(
        ["Current House"],
        [predicted_price]
    )

    ax1.set_ylabel("Predicted Price (Rs.)")
    ax1.set_title(
        "AI/ML Predicted House Price"
    )

    # Price label on bar
    ax1.text(
        0,
        predicted_price,
        f"Rs. {predicted_price:,.0f}",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

    st.pyplot(fig1)

    plt.close(fig1)

    st.caption(
        "This graph is generated dynamically from your current "
        "house prediction."
    )


    # =====================================================
    # GRAPH 2 — PRICE DISTRIBUTION + CURRENT PREDICTION
    # =====================================================

    st.subheader("💰 House Price Distribution")

    fig2, ax2 = plt.subplots(figsize=(9, 5))

    # Dataset price distribution
    ax2.hist(
        df["price"],
        bins=20,
        alpha=0.7
    )

    # Current prediction line
    ax2.axvline(
        predicted_price,
        linestyle="--",
        linewidth=2,
        label=f"Current Prediction: Rs. {predicted_price:,.0f}"
    )

    ax2.set_xlabel("House Price (Rs.)")
    ax2.set_ylabel("Number of Houses")

    ax2.set_title(
        "House Price Distribution with Current Prediction"
    )

    ax2.legend()

    st.pyplot(fig2)

    plt.close(fig2)

    st.caption(
        "The vertical line shows where your current AI/ML "
        "prediction falls within the dataset."
    )


else:

    st.info(
        "🔮 Make a house price prediction first to generate "
        "the dynamic graphs."
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer">
        Built with Python, Pandas, Scikit-learn, Matplotlib & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)