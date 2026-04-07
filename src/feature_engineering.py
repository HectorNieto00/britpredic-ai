# src/feature_engineering.py

import pandas as pd

def create_team_rolling_features_causal(df, window=5):
    """
    Compute rolling team features in a causal way:
    only past matches (from previous seasons + previous matches in same season).
    """
    # Sort by date first
    df = df.sort_values('Date').reset_index(drop=True)

    # Prepare the output DataFrame
    df_features = df.copy()

    # Initialize rolling columns
    df_features['HomeFormPoints'] = 0.0
    df_features['HomeFormGoalsFor'] = 0.0
    df_features['HomeFormGoalsAgainst'] = 0.0
    df_features['AwayFormPoints'] = 0.0
    df_features['AwayFormGoalsFor'] = 0.0
    df_features['AwayFormGoalsAgainst'] = 0.0

    # Store past matches per team
    team_history = {}

    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']

        # Home rolling features
        if home in team_history and len(team_history[home]) >= 1:
            last_matches = pd.DataFrame(team_history[home])
            df_features.at[idx, 'HomeFormPoints'] = last_matches['Points'].tail(window).mean()
            df_features.at[idx, 'HomeFormGoalsFor'] = last_matches['GoalsFor'].tail(window).mean()
            df_features.at[idx, 'HomeFormGoalsAgainst'] = last_matches['GoalsAgainst'].tail(window).mean()
        else:
            df_features.at[idx, 'HomeFormPoints'] = 0.0
            df_features.at[idx, 'HomeFormGoalsFor'] = 0.0
            df_features.at[idx, 'HomeFormGoalsAgainst'] = 0.0

        # Away rolling features
        if away in team_history and len(team_history[away]) >= 1:
            last_matches = pd.DataFrame(team_history[away])
            df_features.at[idx, 'AwayFormPoints'] = last_matches['Points'].tail(window).mean()
            df_features.at[idx, 'AwayFormGoalsFor'] = last_matches['GoalsFor'].tail(window).mean()
            df_features.at[idx, 'AwayFormGoalsAgainst'] = last_matches['GoalsAgainst'].tail(window).mean()
        else:
            df_features.at[idx, 'AwayFormPoints'] = 0.0
            df_features.at[idx, 'AwayFormGoalsFor'] = 0.0
            df_features.at[idx, 'AwayFormGoalsAgainst'] = 0.0

        # Update team history
        home_stats = {'Points': 3 if row['FTR']=='H' else 1 if row['FTR']=='D' else 0,
                      'GoalsFor': row['FTHG'],
                      'GoalsAgainst': row['FTAG']}
        away_stats = {'Points': 3 if row['FTR']=='A' else 1 if row['FTR']=='D' else 0,
                      'GoalsFor': row['FTAG'],
                      'GoalsAgainst': row['FTHG']}

        if home not in team_history:
            team_history[home] = []
        if away not in team_history:
            team_history[away] = []

        team_history[home].append(home_stats)
        team_history[away].append(away_stats)

    return df_features

def create_target(df):
    """Encode match result for ML"""
    df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    return df

def feature_engineering(df, window=5):
    """
    Safe causal feature engineering pipeline.
    """
    df_features = create_team_rolling_features_causal(df, window)
    df_features = create_target(df_features)

    # Drop first matches with zero rolling (optional)
    df_features = df_features.reset_index(drop=True)
    return df_features