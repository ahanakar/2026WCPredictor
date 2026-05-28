import os
import json
import pandas as pd
import numpy as np

def run_feature_engineering():
    print("Starting feature engineering pipeline")

    results_path = "../data/raw/results.csv"
    rankings_path = "../data/raw/fifa_ranking.csv"
    shootouts_path = "../data/raw/shootouts.csv"

    for path in [results_path, rankings_path, shootouts_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required data {path}")
        
    results = pd.read_csv(results_path, parse_dates = ['date'])
    rankings = pd.read_csv(rankings_path, parse_dates = ['rank_date'])
    shootouts = pd.read_csv(shootouts_path, parse_dates = ['date'])

    shootouts['match_id'] = (shootouts['date'].dt.strftime("%Y-%m-%d") + "_" + shootouts['home_team'] + "_" + shootouts['away_team'])
    shootouts_lookup = dict(zip(shootouts['match_id'], shootouts['winner']))

    shootouts_count = {}
    shootouts_wins = {}

    for _, row in shootouts.iterrows():
        home = row['home_team']
        away = row['away_team']
        winner = row['winner']

        shootouts_count[home] = shootouts_count.get(home, 0) + 1
        shootouts_count[away] = shootouts_count.get(away, 0) + 1
        shootouts_wins[winner] = shootouts_wins.get(winner, 0) + 1

    shootouts_tendency = {}
    for team in set(shootouts_count.keys()):
        wins = shootouts_wins.get(team, 0)
        total = shootouts_count.get(team, 0)
        shootouts_tendency[team] = round(wins / total, 2)

    os.makedirs("../data/processed", exist_ok = True)
    with open('../data/processed/shootouts_lookup.json', 'w') as f:
        json.dump(shootouts_lookup, f)
    with open('../data/processed/shootouts_tendency.json', 'w') as f:
        json.dump(shootouts_tendency, f)
    print("Shootout lookup JSONs saved")

    name_cleaner = {
        'IR Iran': 'Iran',
        'Korea Republic': 'South Korea',
        'Czechia': 'Czech Republic',
        'USA': 'United States',
        "Côte d'Ivoire": 'Ivory Coast',
        'Congo DR': 'DR Congo',
        'Cabo Verde': 'Cape Verde',
        'Curaçao': 'Curacao',
        'Türkiye': 'Turkey'
    }
    rankings = rankings.rename(columns = {'country_full': 'team', 'rank': 'fifa_rank', 'rank_date': 'date'})
    rankings['team'] = rankings['team'].replace(name_cleaner)
    results['home_team'] = results['home_team'].replace({'Curaçao': 'Curacao'})
    results['away_team'] = results['away_team'].replace({'Curaçao': 'Curacao'})

    print("Generating historical Elo ratings")

    tournament_weights = {
        'FIFA World Cup': 4.0,
        'UEFA Euro': 3.0,
        'Copa América': 3.0,
        'African Cup of Nations': 3.0,
        'AFC Asian Cup': 2.5,
        'FIFA World Cup qualification': 2.5,
        'UEFA Euro qualification': 2.0,
        'Friendly': 1.0
    }
    elo_dict = {}
    home_elos = []
    away_elos = []
    results_sorted = results.sort_values('date').copy()

    for _, row in results_sorted.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_elo = elo_dict.get(home, 1500.0)
        away_elo = elo_dict.get(away, 1500.0)
        home_elos.append(home_elo)
        away_elos.append(away_elo)

        if pd.isna(row['home_score']) or pd.isna(row['away_score']):
            continue

        home_adv = 0 if row['neutral'] else 75
        dr_home = home_elo - away_elo + home_adv
        W_e_home = 1 / (10 ** (-dr_home / 400) + 1)
        W_e_away = 1 - W_e_home

        if row['home_score'] > row['away_score']:
            W_home, W_away = 1.0, 0.0
        elif row['away_score'] > row['home_score']:
            W_away, W_home = 1.0, 0.0
        else:
            W_home, W_away = 0.5, 0.5
        
        t_weight = tournament_weights.get(row['tournament'], 1.5)
        base_k = 15.0 * t_weight

        goal_diff = abs(row['home_score'] - row['away_score'])
        if goal_diff <= 1:
            g_multiplier = 1.0
        elif goal_diff == 2:
            g_multiplier = 1.5
        else:
            g_multiplier = (11 + goal_diff) / 8

        K = base_k * g_multiplier 
        elo_dict[home] = home_elo + K * (W_home - W_e_home)
        elo_dict[away] = away_elo + K * (W_away - W_e_away)

    results_sorted['home_elo'] = home_elos
    results_sorted['away_elo'] = away_elos
    results_sorted['elo_diff'] = results_sorted['home_elo'] - results_sorted['away_elo']

    results_modern = results_sorted[results_sorted['date'].dt.year>= 2006].copy()
    rankings_modern = rankings[rankings['date'] >= '2005-06-01'].copy()

    print("Merging FIFA rankings into timeline")

    results_modern = results_modern.sort_values(['date'])
    rankings_modern = rankings_modern.sort_values(['date'])
    results_modern = pd.merge_asof(
        results_modern,
        rankings_modern[['date', 'team', 'fifa_rank']],
        on = 'date',
        left_by = 'home_team',
        right_by = 'team', 
        direction = 'backward'
    ).rename(columns = {'fifa_rank': 'home_rank'}).drop(columns = ['team'], errors = 'ignore')

    results_modern = pd.merge_asof(
        results_modern, 
        rankings_modern[['date', 'team', 'fifa_rank']],
        on = 'date', 
        left_by = 'away_team', 
        right_by = 'team', 
        direction = 'backward'
    ).rename(columns = {'fifa_rank': 'away_rank'}).drop(columns = ['team'], errors = 'ignore')

    results_modern['home_rank'] = results_modern['home_rank'].fillna(150)
    results_modern['away_rank'] = results_modern['away_rank'].fillna(150)
    results_modern['rank_diff'] = results_modern['home_rank'] - results_modern['away_rank']

    print("Constructing rolling forms")

    df_home = results_modern[['date', 'home_team', 'home_score', 'away_score']].copy().rename(
        columns = {'home_team': 'team', 'home_score': 'goals_scored', 'away_score': 'goals_conceded'}
    )
    df_home['is_home'] = 1

    df_away = results_modern[['date', 'away_team', 'away_score', 'home_score']].copy().rename(
        columns = {'away_team': 'team', 'away_score': 'goals_scored', 'home_score': 'goals_conceded'}
    )
    df_away['is_home'] = 0

    team_timeline = pd.concat([df_home, df_away]).sort_values(['team', 'date'])
    team_timeline['is_win'] = (team_timeline['goals_scored'] > team_timeline['goals_conceded']).astype(float)
    team_timeline.loc[team_timeline['goals_scored'].isna(), 'is_win'] = np.nan

    team_timeline['rolling_goals_scored'] = team_timeline.groupby('team')['goals_scored'].transform(
        lambda x: x.shift(1).rolling(window = 5, min_periods = 1).mean()
    )
    team_timeline['rolling_goals_conceded'] = team_timeline.groupby('team')['goals_conceded'].transform(
        lambda x: x.shift(1).rolling(window = 5, min_periods = 1).mean()
    )
    team_timeline['rolling_win_rate'] = team_timeline.groupby('team')['is_win'].transform(
        lambda x: x.shift(1).rolling(window = 5, min_periods = 1).mean()
    )

    team_timeline['rolling_goals_scored'] = team_timeline.groupby('team')['rolling_goals_scored'].ffill()
    team_timeline['rolling_goals_conceded'] = team_timeline.groupby('team')['rolling_goals_conceded'].ffill()
    team_timeline['rolling_win_rate'] = team_timeline.groupby('team')['rolling_win_rate'].ffill()
    
    stats_df = team_timeline[['date', 'team', 'rolling_goals_scored', 'rolling_goals_conceded', 'rolling_win_rate']].copy()
    stats_df = stats_df.drop_duplicates(subset=['date', 'team'])
    home_stats = stats_df.rename(columns =
        {'team': 'home_team', 'rolling_goals_scored': 'home_rolling_goals_scored',
        'rolling_goals_conceded': 'home_rolling_goals_conceded', 'rolling_win_rate': 'home_rolling_win_rate'
    })
    away_stats = stats_df.rename(columns =
        {'team': 'away_team', 'rolling_goals_scored': 'away_rolling_goals_scored',
        'rolling_goals_conceded': 'away_rolling_goals_conceded', 'rolling_win_rate': 'away_rolling_win_rate'
    })
    results_modern = results_modern.sort_values('date')
    results_modern = pd.merge(results_modern, home_stats, on = ['date', 'home_team'], how = 'left')
    results_modern = pd.merge(results_modern, away_stats, on = ['date', 'away_team'], how = 'left')

    pre_2022_baseline = results_modern[results_modern['date'].dt.year < 2022]
    avg_home_scored = pre_2022_baseline['home_score'].dropna().mean()
    avg_away_scored = pre_2022_baseline['away_score'].dropna().mean()
    results_modern['home_rolling_win_rate'] = results_modern['home_rolling_win_rate'].fillna(0.33)
    results_modern['away_rolling_win_rate'] = results_modern['away_rolling_win_rate'].fillna(0.33)
    results_modern['home_rolling_goals_scored'] = results_modern['home_rolling_goals_scored'].fillna(avg_home_scored)
    results_modern['away_rolling_goals_scored'] = results_modern['away_rolling_goals_scored'].fillna(avg_away_scored)
    results_modern['home_rolling_goals_conceded'] = results_modern['home_rolling_goals_conceded'].fillna(avg_away_scored)
    results_modern['away_rolling_goals_conceded'] = results_modern['away_rolling_goals_conceded'].fillna(avg_home_scored)

    print("Applying importance weight matrix")

    results_modern['tournament_weight'] = results_modern['tournament'].map(tournament_weights).fillna(1.5)
    current_date = pd.Timestamp.now()
    results_modern['days_since'] = (current_date - results_modern['date']).dt.days
    results_modern['recency_weight'] = np.exp(-results_modern['days_since'] / (4*365 + 1))
    results_modern['match_weight']  = results_modern['tournament_weight'] * results_modern['recency_weight']

    print("Engineering expected goals metrics")

    results_modern['home_attack_strength'] = results_modern['home_rolling_goals_scored'] / avg_home_scored
    results_modern['away_defense_weakness'] = results_modern['away_rolling_goals_conceded'] / avg_home_scored
    results_modern['exp_home_goals'] = results_modern['home_attack_strength'] * results_modern['away_defense_weakness'] * avg_home_scored
    results_modern['away_attack_strength'] = results_modern['away_rolling_goals_scored'] / avg_away_scored
    results_modern['home_defense_weakness'] = results_modern['home_rolling_goals_conceded'] / avg_away_scored
    results_modern['exp_away_goals'] = results_modern['away_attack_strength'] * results_modern['home_defense_weakness'] * avg_away_scored

    results_modern = results_modern.drop(columns = ['tournament', 'tournament_weight', 'recency_weight', 'city', 'country'], errors = 'ignore')
    results_modern['neutral'] = results_modern['neutral'].astype(int)

    print("Splitting data")

    all_non_null_matches = results_modern.dropna(subset=['home_score', 'away_score']).copy()
    future_fixtures = results_modern[results_modern['home_score'].isnull()].copy()
    backtest_train = results_modern[results_modern['date'].dt.year < 2022].dropna(subset=['home_score', 'away_score']).copy()
    backtest_test = all_non_null_matches[all_non_null_matches['date'].dt.year >= 2022].copy()
    main_train = all_non_null_matches.copy()
    main_test = future_fixtures.copy()

    backtest_train.to_csv("../data/processed/backtest_train.csv", index = False)
    backtest_test.to_csv("../data/processed/backtest_test.csv", index = False)
    main_train.to_csv("../data/processed/main_train.csv", index = False)
    if not main_test.empty:
        main_test.to_csv("../data/processed/main_test_groupstage.csv", index = False)
        print(f"Exported {len(main_test)} fixtures to main_test_groupstage.csv")
    else:
        print("No null values found to assign to main_test_groupstage.csv")

    print("Pipeline completed")
    print(f"Backtest track: Train size -- {len(backtest_train)} | Test size -- {len(backtest_test)}")
    print(f"Main training size: {len(main_train)}")

if __name__ == "__main__":
    run_feature_engineering()