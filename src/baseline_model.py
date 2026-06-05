import os
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import d2_tweedie_score
from scipy.stats import poisson

def load_data():
    print("Loading processed datasets")
    train_path = "../data/processed/backtest_train.csv"
    test_path = "../data/processed/backtest_test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    features_to_drop = [
        'date', 'home_team', 'away_team', 'home_score', 'away_score', 'exp_home_goals', 'exp_away_goals', 'match_weight', 'days_since'
    ]
    X_train = train.drop(columns = features_to_drop, errors = 'ignore')
    X_val = test.drop(columns = features_to_drop, errors = 'ignore')
    y_train_home = train['home_score']
    y_val_home = test['home_score']
    y_train_away = train['away_score']
    y_val_away = test['away_score']
    w_train = train['match_weight']
    w_val = test['match_weight']

    return X_train, X_val, y_train_home, y_val_home, y_train_away, y_val_away, w_train, w_val, test

def run_baseline_poisson():
    X_train, X_val, y_train_home, y_val_home, y_train_away, y_val_away, w_train, w_val, test_df = load_data()
    print("Training baseline linear Poisson regressors")

    baseline_home = PoissonRegressor(alpha = 1.0, max_iter = 1000)
    baseline_away = PoissonRegressor(alpha = 1.0, max_iter = 1000)

    baseline_home.fit(X_train, y_train_home, sample_weight = w_train)
    baseline_away.fit(X_train, y_train_away, sample_weight = w_train)

    pred_home_lambdas = baseline_home.predict(X_val)
    pred_away_lambdas = baseline_away.predict(X_val)

    home_d2 = d2_tweedie_score(y_val_home, pred_home_lambdas, power = 1)
    away_d2 = d2_tweedie_score(y_val_away, pred_away_lambdas, power = 1)

    print(f"baseline goal fits: home d2 score {home_d2:.4f} | away d2 score {away_d2:.4f}")

if __name__ == "__main__":
    run_baseline_poisson()