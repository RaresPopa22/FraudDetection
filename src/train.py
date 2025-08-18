import argparse
import os
import yaml

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

def train_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data = pd.read_csv(config['data_paths']['raw_data'])

    data['Hour'] = data['Time'].apply(lambda x: np.floor(x / 3600))
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour']/23.0)
    data['Hour_cos'] = np.sin(2 * np.pi * data['Hour']/23.0)
    data.drop('Hour', axis=1, inplace=True)
    data.drop('Time', axis=1, inplace=True)

    X = data.drop('Class', axis=1)
    y = data['Class']

    print(f'X.shape={X.shape}')
    print(f'y.shape={y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['preprocessing']['test_size'], random_state=1, stratify=y)

    scaler = StandardScaler()
    amount_col = config['preprocessing']['amount_col']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scaler.fit(X_train_scaled[[amount_col]])

    X_train_scaled[amount_col] = scaler.transform(X_train_scaled[[amount_col]])
    X_test_scaled[amount_col] = scaler.transform(X_test_scaled[[amount_col]])

    model_name = config['model_name']
    params = config['model_params']

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    X_train_scaled, X_train_eval, y_train, y_train_eval = train_test_split(
        X_train_scaled, y_train, test_size=0.15, random_state=1, stratify=y_train
    )

    fit_params={}
    early_stopping_config = config.get('early_stopping', {})

    if early_stopping_config.get('enabled', False):
        rounds = early_stopping_config['stopping_rounds']
        eval_set = [(X_train_eval, y_train_eval)]
        print(f"Early stopping enabled with {rounds} rounds")

    if model_name == 'lightgbm':
        params['scale_pos_weight'] = scale_pos_weight
        model = lgb.LGBMClassifier(**params, random_state=1, n_jobs=-1)

        if early_stopping_config.get('enabled', False):
            fit_params['callbacks'] = [lgb.early_stopping(stopping_rounds=rounds, verbose=False)]
            fit_params['eval_set'] = eval_set
    elif model_name == 'xgboost':
        params['scale_pos_weight'] = scale_pos_weight
        if early_stopping_config.get('enabled', False):
            params['early_stopping_rounds'] = rounds
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = False

        model = xgb.XGBClassifier(**params, random_state=1, n_jobs=-1, eval_metric='logloss')


    else:
        raise ValueError(f"Model '{model_name}' not supported")

    model.fit(X_train_scaled, y_train, **fit_params)

    os.makedirs(config['model_output_paths']['dir'], exist_ok=True)
    joblib.dump(model, config['model_output_paths']['model'])
    joblib.dump(scaler, config['model_output_paths']['scaler'])
    X_test_scaled.to_csv(config['data_paths']['X_test_scaled'], index=False)
    y_test.to_csv(config['data_paths']['y_test'], index=False)

    print(f"Model, scaler and test data saved successfully to {config['model_output_paths']['model']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()

    train_model(args.config)
