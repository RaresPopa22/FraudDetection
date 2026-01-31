import logging
import os

import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pos_weight(y_train):
    return y_train.value_counts()[0] / y_train.value_counts()[1]


def get_class_weight(y_train):
    return {0: y_train.value_counts()[0], 1: y_train.value_counts()[1]}

def get_fit_params(model_name, config, X_train_eval, y_train_eval):
    logger.info(f'Setting up fit params...')
    early_stopping_config = config.get('early_stopping', {})
    fit_params = {}

    if early_stopping_config.get('enabled', False):
        rounds = early_stopping_config['stopping_rounds']
        fit_params['eval_set'] = [(X_train_eval, y_train_eval)]

        if model_name == 'lightgbm':
            fit_params['eval_metric'] = 'average_precision'
            fit_params['callbacks'] = [lgb.early_stopping(stopping_rounds=rounds, verbose=False)]
        elif model_name == 'xgboost':
            fit_params['verbose'] = False

    return fit_params


def init_lightgbm(params):
    return lgb.LGBMClassifier(**params, random_state=1, n_jobs=-1, metric='average_precision', verbosity=-1)


def init_xgboost(params, pos_weight, config):
    params['scale_pos_weight'] = pos_weight

    early_stopping_config = config.get('early_stopping', {})
    if early_stopping_config.get('enabled', False):
        rounds = early_stopping_config.get('stopping_rounds', 50)
        params['early_stopping_rounds'] = rounds

    return xgb.XGBClassifier(**params, random_state=1, n_jobs=-1, eval_metric='aucpr')


def init_random_forest(params, pos_weight):
    return RandomForestClassifier(**params, n_jobs=-1, random_state=1, class_weight=pos_weight)


def init_catboost(params, pos_weight, config):
    if config['early_stopping']['enabled']:
        rounds = config['early_stopping']['stopping_rounds']
        params['early_stopping_rounds'] = rounds
    return CatBoostClassifier(**params, scale_pos_weight=pos_weight, eval_metric='PRAUC', random_seed=1, verbose=False)


def init_model(y_train, config):
    model_name = config['model_name']
    logger.info(f'Creating the model {model_name}')
    params = config['model_params']

    if model_name == 'lightgbm':
        return init_lightgbm(params)
    elif model_name == 'xgboost':
        pos_weight = get_pos_weight(y_train)
        return init_xgboost(params, pos_weight, config)
    elif model_name == 'random_forest':
        class_weight = get_class_weight(y_train)
        return init_random_forest(params, class_weight)
    elif model_name == 'catboost':
        pos_weight = get_pos_weight(y_train)
        return init_catboost(params, pos_weight, config)
    else:
        raise ValueError(f"Model '{model_name}' not supported")


def save_model(config, model):
    os.makedirs(config['model_output_paths']['dir'], exist_ok=True)
    joblib.dump(model, config['model_output_paths']['model'])
    logger.info(f"Model saved successfully to {config['model_output_paths']['model']}")
