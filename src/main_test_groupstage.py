import os
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from scipy.stats import poisson
from advanced_model import run_tuning

def load_data():
    print("loading required data")
    train_path = '../data/processed/main_train.csv'
    groupstage_path = '../data/processed/main_test_groupstage.csv'
    train_df = pd.read_csv(train_path)
    groupstage_df = pd.read_csv(groupstage_path)

    features_to_drop = [
        'date', 'home_team', 'away_team', 'home_score', 'away_score', 'exp_home_goals', 'exp_away_goals', 'match_weight', 'days_since'
    ]
    X_train = train_df.drop(columns = features_to_drop, errors = 'ignore')
    y_train_home = train_df['home_score']
    y_train_away = train_df['away_score']
    w_train = train_df['match_weight']
    X_groupstage = groupstage_df.drop(columns = features_to_drop, errors = 'ignore')

    return X_train, y_train_home, y_train_away, w_train, X_groupstage, groupstage_df

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

def generate_predictions():
    X_train, y_train_home, y_train_away, w_train, X_groupstage, groupstage_df = load_data()
    xgb_home_params, xgb_away_params, lgb_home_params, lgb_away_params = run_tuning()

    print("training final production mdoels")
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

    print("simulating 2026 groupstage")
    results = []
    xgb_h_lambdas = xgb_home.predict(X_groupstage)
    xgb_a_lambdas = xgb_away.predict(X_groupstage)
    lgb_h_lambdas = lgb_home.predict(X_groupstage)
    lgb_a_lambdas = lgb_away.predict(X_groupstage)

    for idx, row in groupstage_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

        final_h_lambda = (0.5 * xgb_h_lambdas[idx]) + (0.5 * lgb_h_lambdas[idx])
        final_a_lambda = (0.5 * xgb_a_lambdas[idx]) + (0.5 * lgb_a_lambdas[idx])

        matrix = get_dynamic_score_matrix(final_h_lambda, final_a_lambda)

        home_win_p = np.sum(np.tril(matrix, -1))
        draw_p = np.sum(np.diag(matrix))
        away_win_p = np.sum(np.triu(matrix, 1))
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        predicted_score = f"{max_idx[0]}-{max_idx[1]}"

        results.append({
            'Home Team': home_team,
            'Away Team': away_team,
            'Pred Home Goals': round(final_h_lambda, 2),
            'Pred Away Goals': round(final_a_lambda, 2),
            'Home Win %': round(home_win_p * 100, 1),
            'Away Win %': round(away_win_p * 100, 1),
            'Draw %': round(draw_p * 100, 1),
            'Predicted Scoreline': predicted_score
        })

    predictions_df = pd.DataFrame(results)
    output_path = "../data/processed/groupstage_predictions_2026.csv"
    predictions_df.to_csv(output_path, index = False)
    print(f"Saved output predictions to {output_path}")

if __name__ == '__main__':
    generate_predictions()