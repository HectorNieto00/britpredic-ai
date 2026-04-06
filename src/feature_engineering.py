# src/feature_engineering.py

import pandas as pd

def create_team_rolling_features(df, window=5):
    """
    Create rolling features for teams using only past matches.
    """
    df = df.sort_values('Date')
    
    # Initialize home & away stats
    df['HomeGoalsFor'] = df['FTHG']
    df['HomeGoalsAgainst'] = df['FTAG']
    df['AwayGoalsFor'] = df['FTAG']
    df['AwayGoalsAgainst'] = df['FTHG']
    df['HomePoints'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['AwayPoints'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 3})
    
    # Combine home & away into long format
    home_df = df[['Date', 'HomeTeam', 'HomeGoalsFor', 'HomeGoalsAgainst', 'HomePoints']].copy()
    home_df.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Points']
    
    away_df = df[['Date', 'AwayTeam', 'AwayGoalsFor', 'AwayGoalsAgainst', 'AwayPoints']].copy()
    away_df.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Points']
    
    team_df = pd.concat([home_df, away_df])
    team_df = team_df.sort_values(['Team', 'Date'])
    
    # Rolling features: only past matches
    team_df['FormPoints'] = team_df.groupby('Team')['Points'] \
        .transform(lambda x: x.shift(1).rolling(window).mean())
    team_df['FormGoalsFor'] = team_df.groupby('Team')['GoalsFor'] \
        .transform(lambda x: x.shift(1).rolling(window).mean())
    team_df['FormGoalsAgainst'] = team_df.groupby('Team')['GoalsAgainst'] \
        .transform(lambda x: x.shift(1).rolling(window).mean())
    
    return team_df

def merge_team_features(df, team_df):
    """
    Merge rolling features back to match-level dataset
    """
    # Merge home features
    df = df.merge(
        team_df,
        left_on=['Date', 'HomeTeam'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={
        'FormPoints': 'HomeFormPoints',
        'FormGoalsFor': 'HomeFormGoalsFor',
        'FormGoalsAgainst': 'HomeFormGoalsAgainst'
    }).drop(columns=['Team', 'GoalsFor', 'GoalsAgainst', 'Points'])
    
    # Merge away features
    df = df.merge(
        team_df,
        left_on=['Date', 'AwayTeam'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={
        'FormPoints': 'AwayFormPoints',
        'FormGoalsFor': 'AwayFormGoalsFor',
        'FormGoalsAgainst': 'AwayFormGoalsAgainst'
    }).drop(columns=['Team', 'GoalsFor', 'GoalsAgainst', 'Points'])
    
    return df

def create_target(df):
    """
    Encode match result for ML
    """
    df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    return df

def feature_engineering(df):
    """
    Safe feature engineering pipeline
    """
    # Step 1: create team rolling features
    team_df = create_team_rolling_features(df)
    
    # Step 2: merge back
    df = merge_team_features(df, team_df)
    
    # Step 3: target variable
    df = create_target(df)
    
    # Drop rows with NaN (first matches)
    df = df.dropna().reset_index(drop=True)
    
    return df