import os
import warnings
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.metrics import d2_tweedie_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
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
    y_train_away = train['away_score']
    y_val_home = test['home_score']
    y_val_away = test['away_score']
    w_train = train['match_weight']
    w_val = test['match_weight']

    return X_train, X_val, y_train_home, y_train_away, y_val_home, y_val_away, w_train, w_val

def objective_xgb(trial, X_train, X_val, y_train, y_val, w_train, w_val):
    params = {
        'objective': 'count:poisson',
        'eval_metric': 'poisson-nloglik',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log = True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step = 20),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log = True),
        'gamma': trial.suggest_float('gamma', 1e-3, 5.0, log = True),
        'random_state': 42,
        'n_jobs': -1
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, sample_weight = w_train, eval_set = [(X_val, y_val)], sample_weight_eval_set = [w_val], verbose = False)
    preds = model.predict(X_val)
    return d2_tweedie_score(y_val, preds, power = 1)

def objective_lgb(trial, X_train, X_val, y_train, y_val, w_train, w_val):
    params = {
        'objective': 'poisson',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log = True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step = 20),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log = True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'random_state': 42, 
        'n_jobs': -1,
        'verbose': -1
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, sample_weight = w_train)
    preds = model.predict(X_val)
    return d2_tweedie_score(y_val, preds, power = 1)

def run_tuning():
    X_train, X_val, y_train_home, y_train_away, y_val_home, y_val_away, w_train, w_val = load_data()

    print("tuning XGBoost for home goals")
    study_xgb_home = optuna.create_study(direction = 'maximize')
    study_xgb_home.optimize(lambda t: objective_xgb(t, X_train, X_val, y_train_home, y_val_home, w_train, w_val), n_trials = 30)

    print("tuning XGBoost for away goals")
    study_xgb_away = optuna.create_study(direction = 'maximize')
    study_xgb_away.optimize(lambda t: objective_xgb(t, X_train, X_val, y_train_away, y_val_away, w_train, w_val), n_trials = 30)

    print("tuning LightGBM for home goals")
    study_lgb_home = optuna.create_study(direction = 'maximize')
    study_lgb_home.optimize(lambda t: objective_lgb(t, X_train, X_val, y_train_home, y_val_home, w_train, w_val), n_trials = 30)

    print("tuning LightGBM for away goals")
    study_lgb_away = optuna.create_study(direction = 'maximize')
    study_lgb_away.optimize(lambda t: objective_lgb(t, X_train, X_val, y_train_away, y_val_away, w_train, w_val), n_trials = 30)

    print("tuning complete")
    print(f"XGBoost peak home d2 score: {study_xgb_home.best_value: .4f}")
    print(f"XGBoost peak away d2 score: {study_xgb_away.best_value: .4f}")
    print(f"LightGBM home peak d2 score: {study_lgb_home.best_value: .4f}")
    print(f"LightGBM away peak d2 score: {study_lgb_away.best_value: .4f}")

    return (study_xgb_home.best_params, study_xgb_away.best_params, study_lgb_home.best_params, study_lgb_away.best_params)

if __name__ == '__main__':
    run_tuning()