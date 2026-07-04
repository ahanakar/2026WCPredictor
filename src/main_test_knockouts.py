import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from scipy.stats import poisson
from advanced_model import run_tuning

with open('../data/processed/shootouts_tendency.json', 'r') as f:
    shootout_tendencies = json.load(f)

def load_data():
    print("Loading required data for the Knockout Phase")
    train_path = '../data/processed/main_train.csv'
    knockout_path = '../data/processed/main_test_ro16.csv'
    
    train_df = pd.read_csv(train_path)
    knockout_df = pd.read_csv(knockout_path)

    features_to_drop = [
        'date', 'home_team', 'away_team', 'home_score', 'away_score', 
        'exp_home_goals', 'exp_away_goals', 'match_weight', 'days_since'
    ]
    X_train = train_df.drop(columns = features_to_drop, errors = 'ignore')
    y_train_home = train_df['home_score']
    y_train_away = train_df['away_score']
    w_train = train_df['match_weight']
    X_knockout = knockout_df.drop(columns = features_to_drop, errors = 'ignore')

    return X_train, y_train_home, y_train_away, w_train, X_knockout, knockout_df

def get_dynamic_score_matrix(home_lambda, away_lambda):
    max_lambda = max(home_lambda, away_lambda)
    max_goals = max(6, int(np.ceil(max_lambda * 3)))

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)

    matrix_sum = np.sum(matrix)
    if matrix_sum > 0:
        matrix = matrix / matrix_sum
    
    return matrix

def calculate_knockout_advancement(home_team, away_team, score_matrix):
    home_win_90 = np.sum(np.tril(score_matrix, -1))
    draw_90 = np.sum(np.diag(score_matrix))
    away_win_90 = np.sum(np.triu(score_matrix, 1))
    
    home_shootout_rate = shootout_tendencies.get(home_team, 0.50)
    away_shootout_rate = shootout_tendencies.get(away_team, 0.50)
    
    total_rate = home_shootout_rate + away_shootout_rate
    home_win_shootout_prob = home_shootout_rate / total_rate
    away_win_shootout_prob = away_shootout_rate / total_rate
    
    home_advance_via_shootout = draw_90 * home_win_shootout_prob
    away_advance_via_shootout = draw_90 * away_win_shootout_prob

    total_home_advance = home_win_90 + home_advance_via_shootout
    total_away_advance = away_win_90 + away_advance_via_shootout
    
    return total_home_advance, total_away_advance, home_advance_via_shootout, away_advance_via_shootout

def generate_predictions():
    X_train, y_train_home, y_train_away, w_train, X_knockout, knockout_df = load_data()
    xgb_home_params, xgb_away_params, lgb_home_params, lgb_away_params = run_tuning()

    print("Training final production models")
    xgb_home_params['objective'] = 'count:poisson'
    xgb_away_params['objective'] = 'count:poisson'
    lgb_home_params['objective'] = 'poisson'
    lgb_away_params['objective'] = 'poisson'
    lgb_home_params['verbose'] = -1
    lgb_away_params['verbose'] = -1

    xgb_home = xgb.XGBRegressor(**xgb_home_params)
    xgb_away = xgb.XGBRegressor(**xgb_away_params)
    lgb_home = LGBMRegressor(**lgb_home_params)
    lgb_away = LGBMRegressor(**lgb_away_params)

    xgb_home.fit(X_train, y_train_home, sample_weight = w_train)
    xgb_away.fit(X_train, y_train_away, sample_weight = w_train)
    lgb_home.fit(X_train, y_train_home, sample_weight = w_train)
    lgb_away.fit(X_train, y_train_away, sample_weight = w_train)

    print("Simulating 2026 Knockout Phase (Round of 16)")
    results = []
    xgb_h_lambdas = xgb_home.predict(X_knockout)
    xgb_a_lambdas = xgb_away.predict(X_knockout)
    lgb_h_lambdas = lgb_home.predict(X_knockout)
    lgb_a_lambdas = lgb_away.predict(X_knockout)

    for idx, row in knockout_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

        final_h_lambda = (0.5 * xgb_h_lambdas[idx]) + (0.5 * lgb_h_lambdas[idx])
        final_a_lambda = (0.5 * xgb_a_lambdas[idx]) + (0.5 * lgb_a_lambdas[idx])

        matrix = get_dynamic_score_matrix(final_h_lambda, final_a_lambda)

        home_win_90 = np.sum(np.tril(matrix, -1))
        draw_90 = np.sum(np.diag(matrix))
        away_win_90 = np.sum(np.triu(matrix, 1))
        
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        predicted_score = f"{max_idx[0]}-{max_idx[1]}"

        total_home_advance, total_away_advance, home_adv_so, away_adv_so = calculate_knockout_advancement(
            home_team, away_team, matrix
        )

        advancing_team = home_team if total_home_advance >= total_away_advance else away_team

        results.append({
            'Home Team': home_team,
            'Away Team': away_team,
            'Pred Home Goals': round(final_h_lambda, 2),
            'Pred Away Goals': round(final_a_lambda, 2),
            'Home Win 90m %': round(home_win_90 * 100, 1),
            'Away Win 90m %': round(away_win_90 * 100, 1),
            'Draw 90m %': round(draw_p := draw_90 * 100, 1),
            'Predicted 90m Score': predicted_score,
            'Home Advance Shootout %': round(home_adv_so * 100, 1),
            'Away Advance Shootout %': round(away_adv_so * 100, 1),
            'Total Home Advance %': round(total_home_advance * 100, 1),
            'Total Away Advance %': round(total_away_advance * 100, 1),
            'Team To Advance': advancing_team
        })

    predictions_df = pd.DataFrame(results)
    output_path = "../data/processed/ro16_predictions_2026.csv"
    predictions_df.to_csv(output_path, index = False)
    print(f"Saved knockout progression predictions to {output_path}")

if __name__ == '__main__':
    generate_predictions()