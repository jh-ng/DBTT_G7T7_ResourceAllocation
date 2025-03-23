#############################
# 0. Import relevant libraries
#############################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime

#############################
# 1. Load and Preprocess Data
#############################

# Load datasets
attributes_df = pd.read_csv("data/Attributes_DataFrames.csv")
daily_df = pd.read_csv("data/Daily_DataFrames.csv")

# Merge datasets on movie title
merged_df = daily_df.merge(attributes_df, left_on="Movie_Title", right_on="Title", how="left")

# Drop redundant title column
merged_df.drop(columns=["Title"], inplace=True)

#############################
# 2. Feature Engineering
#############################

# Convert Date to datetime object
merged_df["Date"] = pd.to_datetime(merged_df["Date"])

# Extract useful time-based features
merged_df["Year"] = merged_df["Date"].dt.year
merged_df["Month"] = merged_df["Date"].dt.month
merged_df["Day"] = merged_df["Date"].dt.day
merged_df["DayOfWeek"] = merged_df["Date"].dt.weekday  # 0=Monday, 6=Sunday

# One-Hot Encode categorical variables
categorical_cols = ["Distributor", "MPAA-Rating"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoded_cats = pd.DataFrame(encoder.fit_transform(merged_df[categorical_cols]))

# Assign column names and reset index
encoded_cats.columns = encoder.get_feature_names_out(categorical_cols)
encoded_cats.index = merged_df.index

# Concatenate encoded categorical features
merged_df = pd.concat([merged_df, encoded_cats], axis=1)

# Process Genres (multi-label encoding)
merged_df["Genres"] = merged_df["Genres"].apply(lambda x: x.split(";") if isinstance(x, str) else [])
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
genres_onehot = mlb.fit_transform(merged_df["Genres"])
genres_df = pd.DataFrame(genres_onehot, columns=mlb.classes_, index=merged_df.index)

# Concatenate genre features
merged_df = pd.concat([merged_df, genres_df], axis=1)

# Drop unused columns
merged_df.drop(columns=["Genres", "Movie_Title", "Date"] + categorical_cols, inplace=True)

#############################
# 3. Train Machine Learning Models
#############################

# Define features / Attribute and target variable / Class
X = merged_df.drop(columns=["Daily"])  # Features/ Attribute
y = merged_df["Daily"]  # Target variable / Class

# Scale numerical features for Linear Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# Compare models
best_model = rf_model if mae_rf < mae_lr else lr_model
selected_model = "Random Forest" if mae_rf < mae_lr else "Linear Regression"

print(f"Random Forest MAE: ${mae_rf:.2f}")
print(f"Linear Regression MAE: ${mae_lr:.2f}")
print(f"Selected Model: {selected_model}")

#############################
# 4. Predict Daily Earnings for a New Movie
#############################

def predict_daily_earnings(movie_details, model, encoder, mlb, scaler):
    """Predict daily earnings for a given movie based on input details."""
    
    # Create DataFrame for new movie
    new_movie_df = pd.DataFrame([movie_details])
    
    # Extract date features
    new_movie_df["Date"] = pd.to_datetime(new_movie_df["Date"])
    new_movie_df["Year"] = new_movie_df["Date"].dt.year
    new_movie_df["Month"] = new_movie_df["Date"].dt.month
    new_movie_df["Day"] = new_movie_df["Date"].dt.day
    new_movie_df["DayOfWeek"] = new_movie_df["Date"].dt.weekday
    
    # Encode categorical features
    encoded_cats = pd.DataFrame(encoder.transform(new_movie_df[categorical_cols]))
    encoded_cats.columns = encoder.get_feature_names_out(categorical_cols)
    
    # Process genres
    new_movie_df["Genres"] = new_movie_df["Genres"].apply(lambda x: x.split(";") if isinstance(x, str) else [])
    genres_onehot = pd.DataFrame(mlb.transform(new_movie_df["Genres"]), columns=mlb.classes_)
    
    # Concatenate features
    new_movie_df = pd.concat([new_movie_df, encoded_cats, genres_onehot], axis=1)
    
    # Drop unused columns
    new_movie_df.drop(columns=["Genres", "Date"] + categorical_cols, inplace=True)
    
    # Align feature columns with training data
    missing_cols = set(X.columns) - set(new_movie_df.columns)
    for col in missing_cols:
        new_movie_df[col] = 0  # Fill missing columns with 0
    
    new_movie_df = new_movie_df[X.columns]  # Ensure same column order
    
    # Scale features
    new_movie_scaled = scaler.transform(new_movie_df)
    
    # Predict earnings
    return model.predict(new_movie_scaled)[0]

# Example Usage
new_movie = {
    "Budget": 100000000,
    "Distributor": "Warner Bros",
    "MPAA-Rating": "PG-13",
    "Runtime": 120,
    "Genres": "Action|Adventure",
    "Theaters": 3500,
    "Date": "2025-06-15"
}

predicted_earnings = predict_daily_earnings(new_movie, best_model, encoder, mlb, scaler)
print(f"Predicted daily earnings: ${predicted_earnings:.2f}")