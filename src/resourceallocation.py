#############################
# 0. Import relevant libraries
#############################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer

#############################
# 1. Load and Preprocess Data
#############################

# Load datasets
attributes_df = pd.read_csv("../data/Attributes_DataFrame.csv", nrows=10000)
daily_df = pd.read_csv("../data/Daily_DataFrame.csv", nrows=10000)

# Merge datasets on movie title
merged_df = daily_df.merge(attributes_df, left_on="Movie_Title", right_on="Title", how="left")

# Drop redundant title column
merged_df.drop(columns=["Title"], inplace=True)

# For numerical columns, fill missing values with the mean
numerical_cols = merged_df.select_dtypes(include=["float64", "int64"]).columns
imputer_num = SimpleImputer(strategy='mean')  # Use 'mean' for numerical data
merged_df[numerical_cols] = imputer_num.fit_transform(merged_df[numerical_cols])

# For categorical columns, fill missing values with the mode (most frequent value)
categorical_cols = merged_df.select_dtypes(include=["object"]).columns
imputer_cat = SimpleImputer(strategy='most_frequent')  # Use 'most_frequent' for categorical data
merged_df[categorical_cols] = imputer_cat.fit_transform(merged_df[categorical_cols])

#############################
# 2. Feature Engineering
#############################

# Convert Date to datetime object
merged_df["Date"] = pd.to_datetime(merged_df["Date"], dayfirst=True)

# Extract useful time-based features
merged_df["Year"] = merged_df["Date"].dt.year
merged_df["Month"] = merged_df["Date"].dt.month
merged_df["Day"] = merged_df["Date"].dt.day
merged_df["DayOfWeek"] = merged_df["Date"].dt.weekday  # 0=Monday, 6=Sunday

# One-Hot Encode categorical variables
categorical_cols = ["Distributor", "MPAA-Rating"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cats = pd.DataFrame(encoder.fit_transform(merged_df[categorical_cols]))

# Assign column names and reset index
encoded_cats.columns = encoder.get_feature_names_out(categorical_cols)
encoded_cats.index = merged_df.index

# Concatenate encoded categorical features
merged_df = pd.concat([merged_df, encoded_cats], axis=1)

# Process Genres (multi-label encoding)
merged_df["Genres"] = merged_df["Genres"].apply(lambda x: x.replace("|", ";").split(";") if isinstance(x, str) else [])
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
r2_rf = r2_score(y_test, y_pred_rf)  # Calculate R-squared for Random Forest

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)  # Calculate R-squared for Linear Regression

# Compare models
best_model = rf_model if r2_rf > r2_lr else lr_model
selected_model = "Random Forest" if r2_rf > r2_lr else "Linear Regression"

print(f"Random Forest R²: {r2_rf:.4f}")
print(f"Linear Regression R²: {r2_lr:.4f}")
print(f"Selected Model: {selected_model}")

# Compare models
best_model = rf_model if r2_rf > r2_lr else lr_model
selected_model = "Random Forest" if r2_rf > r2_lr else "Linear Regression"

print(f"Random Forest R²: {r2_rf:.4f}")
print(f"Linear Regression R²: {r2_lr:.4f}")
print(f"Selected Model: {selected_model}")

#############################
# 4. Predict Daily Earnings for a New Movie within a given data range
#############################

import pandas as pd

def predict_daily_earnings_range(movie_details, start_date, end_date, model, encoder, mlb, scaler):
    """Predict daily earnings for each day within a given date range."""
    
    # Convert string dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # List to store predictions for each date
    predictions = []
    
    # Loop through each date in the range
    for single_date in pd.date_range(start_date, end_date):
        # Copy movie details and update the date
        movie_details_copy = movie_details.copy()
        movie_details_copy["Date"] = single_date
        
        # Create DataFrame for new movie
        new_movie_df = pd.DataFrame([movie_details_copy])
        
        # Extract date features
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
        
        # Predict earnings for this date
        daily_earnings = model.predict(new_movie_scaled)[0]
        
        # Store prediction along with the date
        predictions.append({"Date": single_date, "Predicted Earnings": daily_earnings})
    
    # Return the predictions
    return pd.DataFrame(predictions)

# Example Usage
new_movie = {
    "Budget": 100000000,
    "Distributor": "Warner Bros",
    "MPAA-Rating": "PG-13",
    "Runtime": 120,
    "Genres": "Action;Adventure",
    "Theaters": 3500,
    "Date": "2025-06-15"
}

start_date = "2025-06-15"
end_date = "2025-06-20"

predictions_df = predict_daily_earnings_range(new_movie, start_date, end_date, best_model, encoder, mlb, scaler)
print(predictions_df)
