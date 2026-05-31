# House Price Prediction using Machine Learning

This is an end-to-end machine learning project that predicts house prices based on property features such as area, bedrooms, bathrooms, location quality, furnishing status, parking spaces, house age, and distance from the city.

## Project Overview

The goal of this project is to build a regression model that can estimate house prices using structured property data.

The project includes data analysis, preprocessing, model training, evaluation, model saving, and a Streamlit web app for interactive predictions.

## Dataset Features

- area_sqft
- bedrooms
- bathrooms
- stories
- parking_spaces
- house_age
- distance_city_km
- property_type
- location_quality
- furnishing_status
- mainroad
- basement
- has_garden
- price

## Workflow

1. Data loading
2. Missing values analysis
3. Duplicate records check
4. Statistical summary
5. Exploratory Data Analysis
6. Outlier analysis
7. Feature-target split
8. Train-test split
9. Pipeline-based preprocessing
10. Model training
11. Model comparison
12. Final model saving
13. Streamlit app development

## Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Best Model

Linear Regression performed best with an R2 score of approximately 0.847.

## Model Performance

- R2 Score: 0.847
- Mean Absolute Error: around 1.21 million
- RMSE: around 6.99 million

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Streamlit App

The project includes a Streamlit web app where users can enter house details and get a predicted house price.

To run the app:

```bash
streamlit run app.py