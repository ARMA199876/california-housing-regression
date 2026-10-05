import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("models/housing_regression.joblib")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="California Housing Prediction",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏠 California Housing Price Prediction")

st.write(
    "Enter the characteristics of a California census block group "
    "to estimate its median house value."
)

st.info(
    "The model predicts the target in units of $100,000. "
    "For example, 2.5 means approximately $250,000."
)


# --------------------------------------------------
# Input features
# --------------------------------------------------

st.subheader("Property Information")

col1, col2 = st.columns(2)

with col1:

    MedInc = st.number_input(
        "Median Income",
        min_value=0.0,
        value=3.5
    )

    HouseAge = st.number_input(
        "House Age",
        min_value=0.0,
        value=25.0
    )

    AveRooms = st.number_input(
        "Average Rooms",
        min_value=0.0,
        value=5.0
    )

    AveBedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.0,
        value=1.0
    )


with col2:

    Population = st.number_input(
        "Population",
        min_value=0.0,
        value=1000.0
    )

    AveOccup = st.number_input(
        "Average Occupancy",
        min_value=0.0,
        value=3.0
    )

    Latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    Longitude = st.number_input(
        "Longitude",
        value=-118.0
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict House Value"):

    input_data = pd.DataFrame(
        [
            {
                "MedInc": MedInc,
                "HouseAge": HouseAge,
                "AveRooms": AveRooms,
                "AveBedrms": AveBedrms,
                "Population": Population,
                "AveOccup": AveOccup,
                "Latitude": Latitude,
                "Longitude": Longitude
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    predicted_value = prediction * 100_000

    st.success("Prediction completed!")

    st.metric(
        "Estimated Median House Value",
        f"${predicted_value:,.0f}"
    )

    st.caption(
        "Educational prediction based on the trained California Housing model. "
        "It is not a guaranteed market price."
    )
