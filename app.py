import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

# Page title
st.set_page_config(page_title="House Price Predictor")

st.title("🏠 House Price Prediction App")

st.write(
    """
    Predict house prices using a Machine Learning model
    trained with Gradient Boosting Regression.
    """
)

st.metric("Model R² Score", "0.7746")

st.subheader("Enter House Details")

# Inputs

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

grade = st.slider(
    "House Grade",
    min_value=1,
    max_value=13,
    value=7
)

has_basement = st.selectbox(
    "Basement",
    [0, 1]
)

living_in_m2 = st.number_input(
    "Living Area (m²)",
    min_value=20,
    max_value=1000,
    value=150
)

renovated = st.selectbox(
    "Renovated",
    [0, 1]
)

nice_view = st.selectbox(
    "Nice View",
    [0, 1]
)

perfect_condition = st.selectbox(
    "Perfect Condition",
    [0, 1]
)

real_bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

has_lavatory = st.selectbox(
    "Has Lavatory",
    [0, 1]
)

single_floor = st.selectbox(
    "Single Floor",
    [0, 1]
)

month = st.slider(
    "Month Sold",
    min_value=1,
    max_value=12,
    value=6
)

quartile_zone = st.selectbox(
    "Quartile Zone",
    [1, 2, 3, 4]
)

# Prediction button
if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "bedrooms": [bedrooms],
        "grade": [grade],
        "has_basement": [has_basement],
        "living_in_m2": [living_in_m2],
        "renovated": [renovated],
        "nice_view": [nice_view],
        "perfect_condition": [perfect_condition],
        "real_bathrooms": [real_bathrooms],
        "has_lavatory": [has_lavatory],
        "single_floor": [single_floor],
        "month": [month],
        "quartile_zone": [quartile_zone]
    })

    prediction = model.predict(input_df)

    st.success(
        f"Estimated House Price: ₹ {prediction[0]:,.2f}"
    )