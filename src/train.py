import argparse
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

from src.data_processing import read_configs, preprocess_data
from src.models.tree_model import get_fit_params, init_model, save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_tree_model(config):
    X_train_scaled, X_test, y_train, y_test, scaler = preprocess_data(config)
    X_train_final, X_train_eval, y_train_final, y_train_eval = train_test_split(
        X_train_scaled, y_train, test_size=config['preprocessing']['eval_size'], random_state=1, stratify=y_train
    )

    model_name = config['model_name']

    if model_name == 'ensemble':
        estimators = []
        for estimator, estimator_config in config['estimators'].items():
            logger.info(f'Setting up estimator: {estimator}')
            estimator_config['model_name'] = estimator
            fit_params = get_fit_params(estimator, estimator_config, X_train_eval, y_train_eval)
            estimator_model = init_model(y_train_final, estimator_config)

            logger.info('Fitting the model...')
            estimator_model.fit(X_train_final, y_train_final, **fit_params)
            logger.info('Finished fit process')

            estimators.append((estimator, estimator_model))
            logger.info('Done.')

        model = VotingClassifier(estimators=estimators, voting=config['voting'], n_jobs=-1)
        model.estimators_ = [estimator_model for estimator_name, estimator_model in estimators]
    else:
        fit_params = get_fit_params(model_name, config, X_train_eval, y_train_eval)
        model = init_model(y_train_final, config)
        model.fit(X_train_final, y_train_final, **fit_params)

        plot_learning_curve(config['plot_learning_curve'], model, config['model_name'])

    save_model(config, model)
    save_test_data(config, X_test, y_test)
    joblib.dump(scaler, config['model_output_paths']['scaler'])


def train_model(base_config, specific_config):
    config = read_configs(base_config, specific_config)

    train_tree_model(config)


def save_test_data(config, X_test, y_test):
    X_test.to_csv(config['data_paths']['X_test'], index=False)
    y_test.to_csv(config['data_paths']['y_test'], index=False)


def plot_learning_curve(enabled, model, model_name):
    if enabled:
        if model_name == 'xgboost':
            results = model.evals_result()
            epochs = len(results['validation_0']['aucpr'])
            x_axis = range(0, epochs)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_axis, results['validation_0']['aucpr'], label='Test/Validation')
            ax.legend()
            plt.ylabel('XGBoost AUC')
            plt.xlabel('Boosting Round')
            plt.title('XGBoost AUC Over Training Rounds')
            plt.grid(True)
        elif model_name == 'lightgbm':
            lgb.plot_metric(model, metric='average_precision')
        elif model_name == 'catboost':
            results = model.get_evals_result()

            plt.figure(figsize=(10, 6))
            plt.plot(results['validation']['PRAUC'])
            plt.xlabel('Boosting Round')
            plt.ylabel('PRAUC')
            plt.title('CatBoost Learning Curve')

        plt.show()


if __name__ == '__main__':
    base_config_path = Path(__file__).parent.parent / 'config' / 'base_config.yaml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()

    train_model(base_config_path, args.config)
