# DBTT_G7T7_ResourceAllocation
# DBTT Cathay Project Movie Recommender

Welcome to the ML Movie Recommender project! This project aims to build a machine learning-based recommender system that predicts the daily earnings of movies based on various attributes like genre, distributor, MPAA rating, and more. The recommender is trained using a combination of **Random Forest** and **Linear Regression** models. The project was developed as part of the DBTT Cathay Project.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Methodology](#model-methodology)
- [Implementation Details](#implementation-details)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

The goal of this project is to create a movie recommender system that:

- Analyzes movie metadata (such as genres, distributor, and MPAA rating) along with historical earnings data.
- Trains machine learning models (Random Forest and Linear Regression) to predict daily earnings for movies.
- Provides a framework for making real-time predictions based on input movie details, such as budget, runtime, and genre.

## Data Description

The project uses two primary datasets:

1. **Attributes_DataFrame.csv**: Contains movie details, including attributes such as budget, distributor, genre, MPAA rating, runtime, and more.
2. **Daily_DataFrame.csv**: Contains daily earnings data for the movies.

These datasets are used to train and evaluate the machine learning models. Additionally, real-time predictions are made using input movie details, such as genre and budget.

## Exploratory Data Analysis (EDA)

The EDA phase was performed to understand the dataset and make informed decisions for model development:

- **Data Quality and Preprocessing**:
  - Merged datasets based on the movie title.
  - Handled missing values by imputing numerical columns with the mean and categorical columns with the most frequent value.
  
- **Feature Engineering**:
  - Extracted time-based features from the date field (year, month, day, and day of the week).
  - One-hot encoded categorical variables such as distributor and MPAA rating.
  - Applied multi-label encoding to the genres field to create binary genre features.

- **Model Training**:
  - Trained and evaluated both Random Forest and Linear Regression models to predict the daily earnings of movies.

## Model Methodology

The model uses a **content-based approach**:

- **Random Forest** and **Linear Regression** are used to predict daily earnings based on the movie’s attributes and historical data.
- The model features include budget, distributor, genre, MPAA rating, runtime, theaters, and date-based features (year, month, etc.).
- The **Random Forest model** is chosen as the best performing model based on evaluation metrics such as Mean Absolute Error (MAE).

## Implementation Details

### Data Loading and Preprocessing

- The datasets are loaded and merged based on movie titles.
- Missing values in numerical columns are filled with the mean, while categorical columns are filled with the mode (most frequent value).
  
### Feature Engineering

- Categorical features like distributor and MPAA rating are one-hot encoded.
- Genre data is processed using **multi-label binarization** to create binary columns for each genre.

### Model Training

- The features are standardized using **StandardScaler** before training.
- The model is trained using **Random Forest** and **Linear Regression** algorithms, and their performance is compared.

### Real-time Predictions

- A function (`predict_daily_earnings_range`) is implemented to predict daily earnings for a given movie within a specified date range.

## How to Run

1. **Install Dependencies**: Ensure you have the required Python packages installed:

    ```bash
    pip install pandas numpy scikit-learn
    ```

2. **Prepare the Data**: Place the datasets (`Attributes_DataFrame.csv` and `Daily_DataFrame.csv`) in a folder named `data/`.

3. **Run the Script**: Execute the main Python script to generate predictions:

    ```bash
    python movierecommender.py
    ```

4. **View the Predictions**: The script will output predictions for a sample movie, including its predicted daily earnings over a given date range.

## Future Work

- **Model Enhancements**: Explore collaborative filtering or hybrid models to improve recommendation accuracy.
- **Dynamic Updates**: Integrate real-time data streaming to update user profiles and movie earnings predictions.
- **User Interface**: Develop a web interface or API for interactive movie recommendations.

## License

This project is open-source and available under the MIT License.

 
