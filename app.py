import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction App")

model_path = os.path.join("outputs", "house_price_linear_regression_pipeline.pkl")
model = joblib.load(model_path)

area_sqft = st.number_input("Area in Square Feet", min_value=100, max_value=50000, value=3500)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=25, value=4)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=15, value=3)
stories = st.number_input("Stories", min_value=0, max_value=8, value=2)
parking_spaces = st.number_input("Parking Spaces", min_value=0, max_value=12, value=2)
house_age = st.number_input("House Age", min_value=0, max_value=180, value=10)
distance_city_km = st.number_input("Distance from City (km)", min_value=0.0, max_value=250.0, value=12.0)

property_type = st.selectbox("Property Type", ["House", "Apartment", "Townhouse", "Villa"])
location_quality = st.selectbox("Location Quality", ["Low", "Medium", "High", "Premium"])
furnishing_status = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
mainroad = st.selectbox("Main Road Access", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
has_garden = st.selectbox("Garden", ["Yes", "No"])

if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "area_sqft": [area_sqft],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "parking_spaces": [parking_spaces],
        "house_age": [house_age],
        "distance_city_km": [distance_city_km],
        "property_type": [property_type],
        "location_quality": [location_quality],
        "furnishing_status": [furnishing_status],
        "mainroad": [mainroad],
        "basement": [basement],
        "has_garden": [has_garden]
    })

    predicted_price = model.predict(input_data)[0]

    st.success(f"Predicted House Price: {predicted_price:,.2f}")
    st.info(f"Approx Price: {predicted_price / 10_000_000:.2f} crore")