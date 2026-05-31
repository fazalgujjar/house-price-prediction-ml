import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("house_price_dataset.csv")
MODEL_PATH = Path("house_price_model.pkl")
METRICS_PATH = Path("house_price_metrics.json")
TARGET = "price"


def evaluate(name, model, X_test, y_test_log):
    log_pred = model.predict(X_test)
    y_pred = np.expm1(log_pred)
    y_test = np.expm1(y_test_log)

    metrics = {
        "model": name,
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "r2": round(r2_score(y_test, y_pred), 4),
    }
    return metrics, y_pred, y_test


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python generate_house_data.py` first.")

    df = pd.read_csv(DATA_PATH)
    print("Original shape:", df.shape)
    print("Missing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("Duplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates()
    df = df[
        (df["area_sqft"].between(300, 10_000))
        & (df["bedrooms"].between(1, 10))
        & (df["bathrooms"].isna() | df["bathrooms"].between(1, 8))
        & (df["stories"].between(1, 5))
        & (df["house_age"].between(0, 100))
        & (df["distance_city_km"].isna() | df["distance_city_km"].between(0, 80))
        & (df["price"].between(800_000, 90_000_000))
    ].copy()
    print("Cleaned shape:", df.shape)

    df["log_area"] = np.log1p(df["area_sqft"])
    df["age_group"] = pd.cut(
        df["house_age"],
        bins=[-1, 5, 20, 100],
        labels=["New", "Moderate", "Old"],
    ).astype(str)
    df["city_distance_group"] = pd.cut(
        df["distance_city_km"],
        bins=[-1, 8, 20, 100],
        labels=["Near", "Mid", "Far"],
    ).astype(str)

    X = df.drop([TARGET, "area_sqft"], axis=1)
    y_log = np.log1p(df[TARGET])

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=0.2,
        random_state=42,
    )

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    linear_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=16,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    linear_model.fit(X_train, y_train_log)
    rf_model.fit(X_train, y_train_log)

    linear_metrics, _, _ = evaluate("Log Linear Regression", linear_model, X_test, y_test_log)
    rf_metrics, _, _ = evaluate("Random Forest Regressor", rf_model, X_test, y_test_log)

    final_model = rf_model if rf_metrics["r2"] >= linear_metrics["r2"] else linear_model
    final_metrics = rf_metrics if final_model is rf_model else linear_metrics

    artifact = {
        "model": final_model,
        "metrics": final_metrics,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    joblib.dump(artifact, MODEL_PATH)

    all_metrics = {
        "linear_regression": linear_metrics,
        "random_forest": rf_metrics,
        "final_model": final_metrics,
    }
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    print("Linear Regression:", linear_metrics)
    print("Random Forest:", rf_metrics)
    print("Final model:", final_metrics)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

