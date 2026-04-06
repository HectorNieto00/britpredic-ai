# src/data_loader.py

# Load and merge all CSVs
import os
import pandas as pd

def load_all_data(dataset_path="dataset"):
    """
    Load all CSV files from the specified dataset path and merge them into a single DataFrame.
    """
    all_data = [] # List to store DataFrames

    # Loop through all season folders
    for season_folder in sorted(os.listdir(dataset_path)):
        season_path = os.path.join(dataset_path, season_folder)
        if not os.path.isdir(season_path):
            continue

        # Determine season name, e.g., "2020-2021"
        season_name = season_folder.replace("season ", "").replace("_", "-")

        # Loop through all CSV files in the season folder
        for csv_file in os.listdir(season_path):
            if csv_file.endswith(".csv"):
                file_path = os.path.join(season_path, csv_file)

                # Determine league based on file name
                league_map = {
                    "E0": "Premier League",
                    "E1": "Championship",
                    "E2": "League 1",
                    "E3": "League 2",
                    "EC": "Conference",
                }
                league_code = os.path.splitext(csv_file)[0] # Get file name without extension
                league_name = league_map.get(league_code, "Unknown League")

                try:
                    df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1')
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

                # Add season and league columns
                df['Div'] = league_code
                df["Season"] = season_name
                df["League"] = league_name

                # Drop completely empty columns
                df.dropna(axis=1, how='all', inplace=True)

                # Append the DataFrame to the list
                all_data.append(df)

    # Concatenate all DataFrames into a single DataFrame
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_df)} rows from all CSV files.")
        return combined_df
    else:
        print("No data loaded. Please check the dataset path and CSV files.")
        return pd.DataFrame() # Return empty DataFrame if no data was loaded
