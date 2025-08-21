import argparse
import matplotlib.pyplot as plt
import lightgbm as lgb

from src.data_processing import read_configs, preprocess_data, apply_SMOTE
from src.models.tree_model import get_fit_params, init_model, get_pos_weight, save_model, tune_with_random_search, \
    setup_params
from sklearn.model_selection import train_test_split


def train_tree_model(config, tune):
    X_train_scaled, X_test, y_train, y_test = preprocess_data(config)
    X_train_final, X_train_eval, y_train_final, y_train_eval = train_test_split(
        X_train_scaled, y_train, test_size=0.15, random_state=1, stratify=y_train
    )

    X_train_resampled, y_train_resampled = apply_SMOTE(X_train_final, y_train_final)

    model_name = config['model_name']
    fit_params = get_fit_params(model_name, config, X_train_eval, y_train_eval)
    model = init_model(get_pos_weight(y_train_resampled), config)

    if tune:
        hyperparams = setup_params(model_name, config['model_params']['tune'])
        model, _ = tune_with_random_search(
            model_name,
            model,
            hyperparams,
            X_train_resampled,
            y_train_resampled,
            [(X_train_eval, y_train_eval)],
            config)
    else:
        model.fit(X_train_resampled, y_train_resampled, **fit_params)
    plot_learning_curve(True, model, config['model_name'])

    save_model(config, model)
    save_test_data(config, X_test, y_test)

def train_model(base_config, specific_config, tune):
    config = read_configs(base_config, specific_config)

    train_tree_model(config, tune)

def save_test_data(config, X_test, y_test):
    X_test.to_csv(config['data_paths']['X_test'], index=False)
    y_test.to_csv(config['data_paths']['y_test'], index=False)

def plot_learning_curve(enabled, model, model_name):
    if enabled:
        if model_name == 'xgboost':
            results = model.evals_result()
            epochs = len(results['validation_0']['auc'])
            x_axis = range(0, epochs)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_axis, results['validation_0']['auc'], label='Test/Validation')
            ax.legend()
            plt.ylabel('XGBoost AUC')
            plt.xlabel('Boosting Round')
            plt.title('XGBoost AUC Over Training Rounds')
            plt.grid(True)
        elif model_name == 'lightgbm':
            lgb.plot_metric(model, metric='auc')

        plt.show()

if __name__ == '__main__':
    base_config = "config/base_config.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--tune', action='store_true', help='Opt in for tuning hyperparameters')
    args = parser.parse_args()

    train_model(base_config, args.config, args.tune)
