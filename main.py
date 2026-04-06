# main.py

from src.data_loader import load_all_data
from src.preprocessing import preprocess_data
from src.feature_engineering import feature_engineering

# 1️. Load raw data
df = load_all_data("dataset")

print("RAW DATA:")
print(df.head())
print(df.shape)

# 2️. Preprocess
df_clean = preprocess_data(df)

print("\nCLEAN DATA:")
print(df_clean.head())
print(df_clean.shape)
print(df_clean.info())

# 3️. Feature Engineering
df_features = feature_engineering(df_clean)
print(df_features.head())
print(df_features.shape)

# download the df in csv format
df.to_csv("combined_data.csv", index=False)
df_clean.to_csv("clean_data.csv", index=False)
df_features.to_csv("features_data.csv", index=False)