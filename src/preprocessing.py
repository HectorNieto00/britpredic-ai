# src/preprocessing.py

import pandas as pd

def preprocess_data(df):
    """
    Clean football dataset: fix dates, handle missing values, and select relevant columns
    """
    # 1. Clean Dates
    # Convert Date column safely
    df['Date'] = pd.to_datetime(
        df['Date'],
        dayfirst=True, # UK format (DD/MM/YYYY)
        errors='coerce' # Invalid dates become NaT
    )

    # Drop rows with invalid dates
    df = df.dropna(subset=['Date'])

    # 2. Handle Missing Values
    # Keep only rows with essential match info
    essential_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    df = df.dropna(subset=essential_cols)

    # Fill numeric columns with 0 
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fill categorical columns with 'Unknown'
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')

    # 3. Select Relevant Columns
    columns_to_keep = [
        'Div', 'Season', 'League', 'Date', 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'FTR', # Full Time Result (H=Home Win, D=Draw, A=Away Win)
        'HTHG', 'HTAG', 'HTR' # Half Time Result (if exists)
    ]
    # Keep only existing columns
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]

    df = df[columns_to_keep]

    # 4. Final Clean-Up
    # Sort by date (useful for time-based models later) - critical for feature engineering
    df = df.sort_values('Date').reset_index(drop=True)

    return df