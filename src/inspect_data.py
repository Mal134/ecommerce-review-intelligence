
import os
import pandas as pd

# Dataset location
DATA_PATH = "data/raw/Womens Clothing E-Commerce Reviews.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check the CSV filename and location."
    )

# Read dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nRating distribution:")
print(df["Rating"].value_counts().sort_index())

print("\nDuplicate rows:", df.duplicated().sum())

print("\n========== INSPECTION COMPLETED ==========")
