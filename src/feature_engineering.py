# src/feature_engineering.py

import pandas as pd

def add_match_points(df):
    """
    Convert match result into points for each team
    """
    df['HomePoints'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['AwayPoints'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 3})
    return df

def create_team_features(df):
    """
    Create rolling features for teams (last 5 matches)
    """
    df = df.sort_values('Date')

    #Home team features
    df['HomeGoalsFor'] = df['FTHG']
    df['HomeGoalsAgainst'] = df['FTAG']

    # Away team features
    df['AwayGoalsFor'] = df['FTAG']
    df['AwayGoalsAgainst'] = df['FTHG']

    # Combine home & away into one table (long format)
    home_df = df[['Date', 'HomeTeam', 'HomeGoalsFor', 'HomeGoalsAgainst', 'HomePoints']].copy()
    home_df.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Points']

    away_df = df[['Date', 'AwayTeam', 'AwayGoalsFor', 'AwayGoalsAgainst', 'AwayPoints']].copy()
    away_df.columns = ['Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Points']

    team_df = pd.concat([home_df, away_df])
    team_df = team_df.sort_values(['Team', 'Date'])

    # Rolling features for last 5 matches
    team_df['FormPoints'] = team_df.groupby('Team')['Points'] \
        .transform(lambda x: x.shift(1).rolling(5).mean())
    
    team_df['FormGoalsFor'] = team_df.groupby('Team')['GoalsFor'] \
        .transform(lambda x: x.shift(1).rolling(5).mean())
    
    team_df['FormGoalsAgainst'] = team_df.groupby('Team')['GoalsAgainst'] \
        .transform(lambda x: x.shift(1).rolling(5).mean())
    
    return df, team_df

def merge_features(df, team_df):
    """
    Merge team features back into match dataset
    """
    # Merge home team features
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

    # Merge away team features
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
    Encode match reuslt for ML
    """
    df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    return df

def feature_engineering(df):
    """
    Full pipeline
    """
    # Step 1: points
    df = add_match_points(df)

    # Step 2: rolling features
    df, team_df = create_team_features(df)

    # Step 3: merge back
    df = merge_features(df, team_df)

    # Step 4: target variable
    df = create_target(df)

    # Drop rows with NaN in first matches
    df = df.dropna()

    return df
