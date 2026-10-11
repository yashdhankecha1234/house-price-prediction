import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 Seattle House Price Predictor")

c1, c2, c3 = st.columns(3)
with c1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 0.5, 8.0, 2.0, step=0.25)
    sqft_living = st.number_input("Living area (sqft)", 300, 12000, 2000)
    sqft_lot = st.number_input("Lot size (sqft)", 500, 100000, 7000)
    floors = st.number_input("Floors", 1.0, 3.5, 1.0, step=0.5)
with c2:
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View (0-4)", 0, 4, 0)
    condition = st.slider("Condition (1-5)", 1, 5, 3)
    grade = st.slider("Grade (1-13)", 1, 13, 7)
    sqft_above = st.number_input("Sqft above ground", 300, 10000, 1500)
with c3:
    sqft_basement = st.number_input("Sqft basement", 0, 5000, 0)
    yr_built = st.number_input("Year built", 1900, 2015, 1990)
    yr_renovated = st.number_input("Year renovated (0 = never)", 0, 2015, 0)
    zipcode = st.number_input("Zipcode", 98001, 98199, 98103)
    lat = st.number_input("Latitude", 47.15, 47.78, 47.56, format="%.4f")
    long = st.number_input("Longitude", -122.52, -121.31, -122.21, format="%.4f")

if st.button("Predict Price"):
    row = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
        "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront,
        "view": view, "condition": condition, "grade": grade,
        "sqft_above": sqft_above, "sqft_basement": sqft_basement,
        "yr_built": yr_built, "yr_renovated": yr_renovated, "zipcode": zipcode,
        "lat": lat, "long": long,
        "sqft_living15": sqft_living, "sqft_lot15": sqft_lot,   # neighbours ka avg: simple default
        "sale_year": 2015, "sale_month": 6,
        "house_age": 2015 - yr_built,
        "was_renovated": int(yr_renovated > 0),
    }
    X_new = pd.DataFrame([row])[columns]
    price = np.expm1(model.predict(X_new)[0])
    st.success(f"Estimated Price: ${price:,.0f}")