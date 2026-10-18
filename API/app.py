import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE_DIR, "Models", "random_forest_model.joblib")
)

pca_models = joblib.load(
    os.path.join(BASE_DIR, "Data", "Feature Engineered", "pca_models.joblib")
)

print("Model and PCA models loaded successfully!")

from fastapi import FastAPI

# Create the FastAPI application
app = FastAPI(title="MakeMyTrip Price Prediction API")

@app.get("/")
def home():
    return {"message": "MakeMyTrip Price Prediction API is running"}


# Define the input fields accepted by the prediction API
from pydantic import BaseModel

class PackageInput(BaseModel):
    package_name: str
    destination: str
    itinerary: str
    places_covered: str
    hotel_details: str
    airline: str
    sightseeing_places_covered: str
    cancellation_rules: str
    hotel_ratings: float
    travel_month: int
    travel_year: int
    travel_day_of_week: int
    package_type: str
    start_city: str


# Load the SentenceTransformer model used for text embeddings
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Text embedding model loaded successfully!")    

# Define the text columns in the same order used during feature engineering
text_columns = [
    "Package Name",
    "Destination",
    "Itinerary",
    "Places Covered",
    "Hotel Details",
    "Airline",
    "Sightseeing Places Covered",
    "Cancellation Rules"
]

print("Text feature configuration loaded!")

# Load the exact feature names expected by the model
x_train = pd.read_csv(
    os.path.join(BASE_DIR, "Data", "Training Data", "x_train.csv")
)

feature_columns = x_train.columns.tolist()

print(f"Expected model features: {len(feature_columns)}")

# Define the 11 structured features used during training
num_features = [
    "Package Type_Standard",
    "Package Type_Premium",
    "Package Type_Luxury",
    "Travel_Month",
    "Package Type_Budget",
    "Package Type_Deluxe",
    "Hotel Ratings",
    "Start City_New Delhi",
    "Start City_Mumbai",
    "Travel_DayOfWeek",
    "Travel_Year"
]

print("Structured feature configuration loaded:", len(num_features))

# Define the price prediction endpoint
# Load feature names once when the API starts
feature_path = os.path.join(
    BASE_DIR, "Data", "Training Data", "x_train.csv"
)

feature_columns = pd.read_csv(
    feature_path, nrows=0
).columns.tolist()

print("Feature columns loaded:", len(feature_columns))

print("First 10 training features:", feature_columns[:10])
print("Last 15 training features:", feature_columns[-15:])


@app.post("/predict")
def predict_price(package: PackageInput):
    # Convert text inputs into embeddings
    text_values = [
        package.package_name,
        package.destination,
        package.itinerary,
        package.places_covered,
        package.hotel_details,
        package.airline,
        package.sightseeing_places_covered,
        package.cancellation_rules
    ]

    text_embeddings = embedding_model.encode(text_values)

    # Reduce each text embedding using its saved PCA model
    pca_features = []

    for i, column in enumerate(text_columns):
        embedding = text_embeddings[i].reshape(1, -1)
        reduced_embedding = pca_models[column].transform(embedding)
        pca_features.extend(reduced_embedding[0])

    # Create the 11 structured features
    structured_data = {
       "Package Type_Standard": int(package.package_type == "Standard"),
       "Package Type_Premium": int(package.package_type == "Premium"),
       "Package Type_Luxury": int(package.package_type == "Luxury"),
       "Travel_Month": package.travel_month,
       "Package Type_Budget": int(package.package_type == "Budget"),
       "Package Type_Deluxe": int(package.package_type == "Deluxe"),
       "Hotel Ratings": package.hotel_ratings,
       "Start City_New Delhi": int(package.start_city == "New Delhi"),
       "Start City_Mumbai": int(package.start_city == "Mumbai"),
       "Travel_DayOfWeek": package.travel_day_of_week,
       "Travel_Year": package.travel_year
    }

    # Combine the features
    input_features = pca_features + list(structured_data.values())

    input_df = pd.DataFrame(
        [input_features],
        columns=feature_columns
    )

    print("Input feature shape:", input_df.shape)

    # Predict the price per person
    predicted_price = model.predict(input_df)[0]

    return {
        "predicted_price_per_person": round(float(predicted_price), 2)
    }