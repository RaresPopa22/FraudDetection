import argparse

from src.data_processing import read_configs, preprocess_data_tree
from src.models.tree_model import get_fit_params, init_model, get_pos_weight, save_model


def train_tree_model(config):
    X_train_scaled, y_train, X_train_eval, y_train_eval, X_test, y_test = preprocess_data_tree(config)
    fit_params = get_fit_params(config['model_name'], config, X_train_eval, y_train_eval)
    model = init_model(config['model_name'], config['model_params'], get_pos_weight(y_train), config)
    model.fit(X_train_scaled, y_train, **fit_params)

    save_model(config, model)
    save_test_data(config, X_test, y_test)

def train_model(base_config, specific_config):
    config = read_configs(base_config, specific_config)

    train_tree_model(config)

def save_test_data(config, X_test, y_test):
    X_test.to_csv(config['data_paths']['X_test'], index=False)
    y_test.to_csv(config['data_paths']['y_test'], index=False)

if __name__ == '__main__':
    base_config = "../config/base_config.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()

    train_model(base_config, args.config)
