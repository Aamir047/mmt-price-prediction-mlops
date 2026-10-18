
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import os


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="MakeMyTrip Price Predictor",
    page_icon="✈️",
    layout="centered"
)


# --------------------------------------------------
# 2. Load the cleaned dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "Cleaned" / "MMT_Cleaned.csv"

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


@st.cache_data
def load_packages(path):
    return pd.read_csv(path)


st.title("✈️ MakeMyTrip Price Predictor")
st.write("Select a travel package to estimate its price per person.")

try:
    df_packages = load_packages(DATA_PATH)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()


# --------------------------------------------------
# 3. Check required dataset columns
# --------------------------------------------------

text_columns = [
    "Package Name",
    "Destination",
    "Itinerary",
    "Places Covered",
    "Hotel Details",
    "Airline",
    "Sightseeing Places Covered",
    "Cancellation Rules",
]

missing_columns = [
    column for column in text_columns
    if column not in df_packages.columns
]

if missing_columns:
    st.error(
        "Required columns are missing from the dataset: "
        + ", ".join(missing_columns)
    )
    st.stop()


# --------------------------------------------------
# 4. Select an actual package
# --------------------------------------------------

package_names = sorted(
    df_packages["Package Name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not package_names:
    st.error("No package names were found in the dataset.")
    st.stop()

selected_package_name = st.selectbox(
    "Select a Travel Package",
    package_names
)

package_row = df_packages.loc[
    df_packages["Package Name"].astype(str)
    == selected_package_name
].iloc[0]


# --------------------------------------------------
# 5. Identify package categories
# --------------------------------------------------

def get_category(row, column, options):
    if column in row.index and pd.notna(row[column]):
        value = str(row[column])

        if value in options:
            return value

    for option in options:
        encoded_column = f"{column}_{option}"

        if encoded_column in row.index:
            value = pd.to_numeric(
                pd.Series([row[encoded_column]]),
                errors="coerce"
            ).iloc[0]

            if pd.notna(value) and value == 1:
                return option

    return None


package_type_from_data = get_category(
    package_row,
    "Package Type",
    ["Standard", "Premium", "Luxury", "Budget", "Deluxe"]
)

start_city_from_data = get_category(
    package_row,
    "Start City",
    ["New Delhi", "Mumbai"]
)


# --------------------------------------------------
# 6. Collect travel preferences
# --------------------------------------------------

st.subheader("Travel Preferences")

col1, col2 = st.columns(2)

with col1:
    start_city_options = ["New Delhi", "Mumbai"]

    default_city_index = (
        start_city_options.index(start_city_from_data)
        if start_city_from_data in start_city_options
        else 0
    )

    start_city = st.selectbox(
        "Starting City",
        start_city_options,
        index=default_city_index
    )

    hotel_ratings = st.selectbox(
        "Hotel Ratings",
        [1, 2, 3, 4, 5]
    )

    travel_month = st.selectbox(
        "Travel Month",
        list(range(1, 13))
    )

with col2:
    travel_day_of_week = st.selectbox(
        "Travel Day of Week (Monday = 0)",
        list(range(7))
    )

    package_type_options = [
        "Standard", "Premium", "Luxury", "Budget", "Deluxe"
    ]

    default_type_index = (
        package_type_options.index(package_type_from_data)
        if package_type_from_data in package_type_options
        else 0
    )

    package_type = st.selectbox(
        "Package Type",
        package_type_options,
        index=default_type_index
    )

    travel_year = st.selectbox(
        "Travel Year",
        [2026, 2027, 2028]
    )


# --------------------------------------------------
# 7. Display selected package details
# --------------------------------------------------

st.subheader("Selected Package Details")

st.write("**Package:**", selected_package_name)
st.write("**Destination:**", package_row["Destination"])
st.write("**Package Type:**", package_type)
st.write("**Starting City:**", start_city)
st.write("**Selected Hotel Rating:**", hotel_ratings)


# --------------------------------------------------
# 8. Predict price through FastAPI
# --------------------------------------------------

if st.button("Predict Package Price", type="primary"):

    # Use actual text from the selected dataset row.
    payload = {
        "package_name": str(package_row["Package Name"]),
        "destination": str(package_row["Destination"]),
        "itinerary": (
            str(package_row["Itinerary"])
            if pd.notna(package_row["Itinerary"]) else ""
        ),
        "places_covered": (
            str(package_row["Places Covered"])
            if pd.notna(package_row["Places Covered"]) else ""
        ),
        "hotel_details": (
            str(package_row["Hotel Details"])
            if pd.notna(package_row["Hotel Details"]) else ""
        ),
        "airline": (
            str(package_row["Airline"])
            if pd.notna(package_row["Airline"]) else ""
        ),
        "sightseeing_places_covered": (
            str(package_row["Sightseeing Places Covered"])
            if pd.notna(package_row["Sightseeing Places Covered"]) else ""
        ),
        "cancellation_rules": (
            str(package_row["Cancellation Rules"])
            if pd.notna(package_row["Cancellation Rules"]) else ""
        ),
        "hotel_ratings": float(hotel_ratings),
        "travel_month": int(travel_month),
        "travel_year": int(travel_year),
        "travel_day_of_week": int(travel_day_of_week),
        "package_type": package_type,
        "start_city": start_city,
    }

    try:
        with st.spinner("Predicting package price..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

        st.success("Prediction completed!")

        st.metric(
            "Estimated Price Per Person",
            f"₹{result['predicted_price_per_person']:,.2f}"
        )

        st.caption(
            "This is a machine-learning estimate, not a live booking quote."
        )

    except requests.RequestException as error:
        st.error(f"Could not connect to the prediction API: {error}")

    except (KeyError, ValueError) as error:
        st.error(f"Could not process the prediction response: {error}")