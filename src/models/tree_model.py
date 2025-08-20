import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
import os

def get_pos_weight(y_train):
    return y_train.value_counts()[0] / y_train.value_counts()[1]

def get_eval(X_train, y_train, config):
    eval_set = []

    if config['early_stopping']['enabled']:
        rounds = config['early_stopping']['stopping_rounds']
        eval_set = [(X_train, y_train)]
        print(f"Early stopping enabled with {rounds} rounds")

    return eval_set

def get_fit_params(model_name, config, X_train_eval, y_train_eval):
    early_stopping_config = config.get('early_stopping', {})
    eval_set = get_eval(X_train_eval, y_train_eval, config)

    fit_params = {}

    if early_stopping_config.get('enabled', False):
        rounds = early_stopping_config['stopping_rounds']
        fit_params['eval_set'] = eval_set

        if model_name == 'lightgbm':
            fit_params['callbacks'] = [lgb.early_stopping(stopping_rounds=rounds, verbose=False)]
        elif model_name == 'xgboost':
            fit_params['verbose'] = False

    return fit_params

def init_lightgbm(params, pos_weight):
    params['scale_pos_weight'] = pos_weight
    return lgb.LGBMClassifier(**params, random_state=1, n_jobs=-1)

def init_xgboost(params, pos_weight, rounds):
    params['scale_pos_weight'] = pos_weight
    params['early_stopping_rounds'] = rounds
    return xgb.XGBClassifier(**params, random_state=1, n_jobs=-1, eval_metric='logloss')

def init_model(model_name, params, pos_weight, config):
    if model_name == 'lightgbm':
        return init_lightgbm(params, pos_weight)
    elif model_name == 'xgboost':
        return init_xgboost(params, pos_weight, config['early_stopping']['stopping_rounds'])
    else:
        raise ValueError(f"Model '{model_name}' not supported")

def save_model(config, model):
    os.makedirs(config['model_output_paths']['dir'], exist_ok=True)
    joblib.dump(model, config['model_output_paths']['model'])
    print(f"Model saved successfully to {config['model_output_paths']['model']}")