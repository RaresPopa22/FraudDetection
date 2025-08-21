import joblib
import lightgbm as lgb
import xgboost as xgb
import os

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def get_pos_weight(y_train):
    return y_train.value_counts()[0] / y_train.value_counts()[1]

def get_fit_params(model_name, config, X_train_eval, y_train_eval):
    early_stopping_config = config.get('early_stopping', {})

    fit_params = {}

    if early_stopping_config.get('enabled', False):
        rounds = early_stopping_config['stopping_rounds']
        fit_params['eval_set'] = [(X_train_eval, y_train_eval)]

        if model_name == 'lightgbm':
            fit_params['eval_metric'] = 'auc'
            fit_params['callbacks'] = [lgb.early_stopping(stopping_rounds=rounds, verbose=False)]
        elif model_name == 'xgboost':
            fit_params['verbose'] = False

    return fit_params

def init_lightgbm(params, pos_weight):
    params['scale_pos_weight'] = pos_weight
    return lgb.LGBMClassifier(**params, random_state=1, n_jobs=-1)

def setup_params(model_name, params):
    tune_params = {}
    if model_name == 'lightgbm':
        tune_params['num_leaves'] = randint(params['num_leaves'][0], params['num_leaves'][1])
        tune_params['max_depth'] = randint(params['max_depth'][0], params['max_depth'][1])
        tune_params['learning_rate'] = uniform(params['learning_rate'][0], params['learning_rate'][1])
    elif model_name == 'xgboost':
        tune_params['max_depth'] = randint(params['max_depth'][0], params['max_depth'][1])
        tune_params['learning_rate'] = uniform(params['learning_rate'][0], params['learning_rate'][1])

    return tune_params

def init_xgboost(params, pos_weight, config):
    params['scale_pos_weight'] = pos_weight
    if config['early_stopping']['enabled']:
        rounds = config['early_stopping']['stopping_rounds']
        params['early_stopping_rounds'] = rounds

    return xgb.XGBClassifier(**params, random_state=1, n_jobs=-1, eval_metric='auc')

def init_model(pos_weight, config):
    model_name = config['model_name']
    params = config['model_params']['fixed']

    if model_name == 'lightgbm':
        return init_lightgbm(params, pos_weight)
    elif model_name == 'xgboost':
        return init_xgboost(params, pos_weight, config)
    else:
        raise ValueError(f"Model '{model_name}' not supported")

def save_model(config, model):
    os.makedirs(config['model_output_paths']['dir'], exist_ok=True)
    joblib.dump(model, config['model_output_paths']['model'])
    print(f"Model saved successfully to {config['model_output_paths']['model']}")

def tune_with_random_search(model_name, model, params, X_train, y_train, eval_set, config):
    random_search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=50,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=1,
        refit=False
    )

    if model_name == 'xgboost':
        random_search.refit = True
        random_search.fit(X_train, y_train, eval_set=eval_set)
        best_model = random_search.best_estimator_
    elif model_name == 'lightgbm':
        rounds = config['early_stopping']['stopping_rounds']
        fit_params = {
            'eval_set': eval_set,
            'eval_metric': 'auc',
            'callbacks': [lgb.early_stopping(stopping_rounds=rounds, verbose=False)]}
        random_search.fit(X_train, y_train)
        best_params = {**model.get_params(), **random_search.best_params_}
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train, **fit_params)

    print(f'Best score: {random_search.best_score_}')
    print(f'Best parameters found: {random_search.best_params_}')

    return best_model, random_search.best_params_