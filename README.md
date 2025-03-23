# DBTT_G7T7_ResourceAllocation

Welcome to the ML Movie Recommender project! This project aims to build a machine learning-based recommender system that predicts the daily earnings of movies based on various attributes like genre, distributor, MPAA rating, domestic earnings, international earnings, rank and more. The recommender is trained using a combination of **Random Forest** and **Linear Regression** models. The project was developed as part of the DBTT Cathay Project.

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

The goal of this project is to predict the daily earnings for a specific movie given its relevant details, allowing cinemas to decide how much resources they can allocate based on these predictions.

Key Features:
- Analyzes movie metadata (such as genre, distributor, MPAA rating, budget, and runtime) along with historical earnings data.
- Builds a predictive model using machine learning (Random Forest and Linear Regression) to forecast daily earnings for movies.
- Uses the model to make real-time predictions based on input movie details, such as budget, runtime, and genre.
- Helps cinemas make more informed decisions about resource allocation for each movie.

## Data Description

The project uses two primary datasets:

1. **Attributes_DataFrame.csv**: Contains movie details, including the following columns:
   - `Title`: Movie title
   - `Domestic`: Domestic earnings
   - `International`: International earnings
   - `Budget`: Movie budget
   - `Distributor`: Movie distributor
   - `MPAA-Rating`: MPAA rating of the movie
   - `Runtime`: Movie runtime (in minutes)
   - `Genres`: Genre(s) of the movie

2. **Daily_DataFrame.csv**: Contains daily earnings data for the movies, including the following columns:
   - `Movie_Title`: Title of the movie
   - `Date`: Date of the earnings record
   - `Daily`: Daily earnings for the movie
   - `Theaters`: Number of theaters showing the movie on the given date
   - `Rank`: Movie's rank based on earnings on that date

## Exploratory Data Analysis (EDA)

The EDA phase was performed to understand the dataset and make informed decisions for model development:

- **Data Loading and Preprocessing**:
  - Loaded the datasets and merged them based on the movie title.
  - Handled missing values by imputing numerical columns with the mean and categorical columns with the most frequent value.
  - Displayed the resulting DataFrame to inspect the data after preprocessing.

- **Feature Engineering and Visualization**:
  - Applied feature engineering to extract time-based features from the date field (year, month, day, and day of the week).
  - One-hot encoded categorical variables such as distributor and MPAA rating.
  - Applied multi-label encoding to the genres field to create binary genre features.
  - Visualized the DataFrame to observe the structure and distribution of data.

- **Data Visualization**:
  - Visualized the relationship between various features to identify trends, such as:
    - Number of movies per distributor
    - Total earnings by year
    - Total earnings by month
    - Total earnings by day of the week
    - Earnings by genre

- **Correlation Analysis**:
  - Conducted a correlation analysis to identify the relationships between different numerical features in the dataset.
  - Visualized the correlation matrix to gain insights into potential predictors for the model.

## Model Methodology

The model uses a **content-based approach**:

- **Random Forest** and **Linear Regression** are used to predict daily earnings based on the movie’s attributes and historical data.
- The model features include budget, distributor, genre, MPAA rating, runtime, theaters, and date-based features (year, month, etc.).
- Both models will be evaluated and compared based on evaluation metrics such as Mean Absolute Error (MAE) to determine which one performs better.
- The better-performing model will be selected for further use.

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

 
