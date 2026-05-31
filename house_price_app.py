import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "house_price_model.pkl"


st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Prediction System")
st.caption("Estimate house price using property, location, and condition features.")


def load_artifact():
    return joblib.load(MODEL_PATH)


try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error("Model not found. Run `python train_house_model.py` first.")
    st.stop()


model = artifact["model"]
metrics = artifact.get("metrics", {})

st.sidebar.header("Property Details")

area_sqft = st.sidebar.number_input("Area (sqft)", min_value=300, max_value=10000, value=2200, step=50)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 4)
bathrooms = st.sidebar.slider("Bathrooms", 1, 8, 3)
stories = st.sidebar.slider("Stories", 1, 5, 2)
parking_spaces = st.sidebar.slider("Parking Spaces", 0, 8, 1)
house_age = st.sidebar.slider("House Age", 0, 100, 10)
distance_city_km = st.sidebar.slider("Distance from City (km)", 0.0, 80.0, 8.0, 0.5)

st.sidebar.header("Property Category")

property_type = st.sidebar.selectbox("Property Type", ["Apartment", "House", "Villa", "Townhouse"])
location_quality = st.sidebar.selectbox("Location Quality", ["Low", "Medium", "High", "Premium"], index=2)
furnishing_status = st.sidebar.selectbox(
    "Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"],
    index=1,
)
mainroad = st.sidebar.selectbox("Main Road Access", ["No", "Yes"], index=1)
basement = st.sidebar.selectbox("Basement", ["No", "Yes"])
has_garden = st.sidebar.selectbox("Garden", ["No", "Yes"])

log_area = np.log1p(area_sqft)
age_group = "New" if house_age <= 5 else "Moderate" if house_age <= 20 else "Old"
city_distance_group = "Near" if distance_city_km <= 8 else "Mid" if distance_city_km <= 20 else "Far"

house = pd.DataFrame(
    [
        {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking_spaces": parking_spaces,
            "house_age": house_age,
            "distance_city_km": distance_city_km,
            "property_type": property_type,
            "location_quality": location_quality,
            "furnishing_status": furnishing_status,
            "mainroad": mainroad,
            "basement": basement,
            "has_garden": has_garden,
            "log_area": log_area,
            "age_group": age_group,
            "city_distance_group": city_distance_group,
        }
    ]
)

st.subheader("Prediction")

col1, col2 = st.columns(2)
col1.metric("Area", f"{area_sqft:,.0f} sqft")
col2.metric("Model R2", metrics.get("r2", "N/A"))

if st.button("Predict House Price", type="primary"):
    log_prediction = model.predict(house)[0]
    price_prediction = np.expm1(log_prediction)

    st.metric("Estimated House Price", f"PKR {price_prediction:,.0f}")

    if price_prediction >= 30_000_000:
        st.info("Premium property estimate. Review location and luxury features carefully.")
    elif price_prediction >= 12_000_000:
        st.success("Mid-to-high value property estimate.")
    else:
        st.warning("Affordable property estimate. Check area, age, and location quality.")

with st.expander("View Model Input"):
    st.dataframe(house, width="stretch")

