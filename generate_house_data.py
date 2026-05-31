from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
DATA_PATH = Path("house_price_dataset.csv")


def build_dataset(rows=1800):
    rng = np.random.default_rng(RANDOM_SEED)

    area_sqft = rng.integers(450, 6500, rows)
    bedrooms = rng.integers(1, 8, rows)
    bathrooms = rng.integers(1, 6, rows)
    stories = rng.integers(1, 4, rows)
    parking_spaces = rng.integers(0, 4, rows)
    house_age = rng.integers(0, 55, rows)
    distance_city_km = rng.uniform(1, 40, rows).round(1)

    property_type = rng.choice(
        ["Apartment", "House", "Villa", "Townhouse"],
        rows,
        p=[0.35, 0.4, 0.12, 0.13],
    )
    location_quality = rng.choice(
        ["Low", "Medium", "High", "Premium"],
        rows,
        p=[0.18, 0.46, 0.27, 0.09],
    )
    furnishing_status = rng.choice(
        ["Unfurnished", "Semi-Furnished", "Furnished"],
        rows,
        p=[0.36, 0.42, 0.22],
    )
    mainroad = rng.choice(["No", "Yes"], rows, p=[0.22, 0.78])
    basement = rng.choice(["No", "Yes"], rows, p=[0.7, 0.3])
    has_garden = rng.choice(["No", "Yes"], rows, p=[0.62, 0.38])

    location_bonus = pd.Series(location_quality).map(
        {"Low": -700_000, "Medium": 0, "High": 1_500_000, "Premium": 3_500_000}
    ).to_numpy()
    property_bonus = pd.Series(property_type).map(
        {"Apartment": -500_000, "House": 400_000, "Villa": 2_500_000, "Townhouse": 700_000}
    ).to_numpy()
    furnishing_bonus = pd.Series(furnishing_status).map(
        {"Unfurnished": 0, "Semi-Furnished": 500_000, "Furnished": 1_100_000}
    ).to_numpy()
    mainroad_bonus = np.where(mainroad == "Yes", 450_000, 0)
    basement_bonus = np.where(basement == "Yes", 650_000, 0)
    garden_bonus = np.where(has_garden == "Yes", 800_000, 0)
    noise = rng.normal(0, 900_000, rows)

    price = (
        1_300_000
        + area_sqft * 7_600
        + bedrooms * 360_000
        + bathrooms * 560_000
        + stories * 320_000
        + parking_spaces * 270_000
        - house_age * 62_000
        - distance_city_km * 95_000
        + location_bonus
        + property_bonus
        + furnishing_bonus
        + mainroad_bonus
        + basement_bonus
        + garden_bonus
        + noise
    )
    price = np.maximum(price, 1_000_000).round().astype(int)

    df = pd.DataFrame(
        {
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking_spaces": parking_spaces,
            "house_age": house_age,
            "distance_city_km": distance_city_km,
            "property_type": property_type,
            "location_quality": location_quality,
            "furnishing_status": furnishing_status,
            "mainroad": mainroad,
            "basement": basement,
            "has_garden": has_garden,
            "price": price,
        }
    )

    missing_cols = ["bathrooms", "parking_spaces", "distance_city_km", "location_quality", "furnishing_status"]
    for col in missing_cols:
        missing_idx = rng.choice(df.index, size=int(rows * 0.04), replace=False)
        df.loc[missing_idx, col] = np.nan

    duplicate_rows = df.sample(45, random_state=RANDOM_SEED)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    outlier_rows = pd.DataFrame(
        [
            {
                "area_sqft": 50_000,
                "bedrooms": 25,
                "bathrooms": 15,
                "stories": 8,
                "parking_spaces": 12,
                "house_age": 180,
                "distance_city_km": 250,
                "property_type": "Villa",
                "location_quality": "Premium",
                "furnishing_status": "Furnished",
                "mainroad": "Yes",
                "basement": "Yes",
                "has_garden": "Yes",
                "price": 250_000_000,
            },
            {
                "area_sqft": 120,
                "bedrooms": 0,
                "bathrooms": 0,
                "stories": 0,
                "parking_spaces": 0,
                "house_age": 160,
                "distance_city_km": 120,
                "property_type": "Apartment",
                "location_quality": "Low",
                "furnishing_status": "Unfurnished",
                "mainroad": "No",
                "basement": "No",
                "has_garden": "No",
                "price": 200_000,
            },
        ]
    )

    return pd.concat([df, outlier_rows], ignore_index=True)


def main():
    df = build_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved dataset: {DATA_PATH}")
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()

