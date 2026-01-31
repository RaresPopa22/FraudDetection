import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def read_configs(base_path, specific_path):
    base_config = read_config(base_path)
    specific_config = read_config(specific_path)

    return {**base_config, **specific_config}


def load_data(path):
    data = pd.read_csv(path)
    return data


def load_test_data(config):
    X_test = pd.read_csv(config['data_paths']['X_test'])
    y_test = pd.read_csv(config['data_paths']['y_test'])

    return X_test, y_test.squeeze()


def feature_engineer(data):
    top_features = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3', 'V7']

    data['Hour'] = data['Time'].apply(lambda x: np.floor(x / 3600))
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data.drop('Hour', axis=1, inplace=True)
    data.drop('Time', axis=1, inplace=True)

    data['V17_V14'] = data['V17'] * data['V14']
    data['V17_V14_hour_sin'] = data['V17_V14'] * data['Hour_sin']
    data['V17_V14_hour_cos'] = data['V17_V14'] * data['Hour_cos']
    data['V12_V10'] = data['V12'] * data['V10']

    data['top_features'] = data['V17_V14'] * data['V12_V10'] * data['V16'] * data['V3'] * data['V7']

    data['V17_squared'] = data['V17'] ** 2
    data['V14_squared'] = data['V14'] ** 2
    data['V12_squared'] = data['V12'] ** 2
    data['top_features_squared'] = data['V17_squared'] * data['V14_squared'] * data['V12_squared'] * data['V10'] ** 2

    data['top_mean'] = data[top_features].mean(axis=1)
    data['top_min'] = data[top_features].min(axis=1)
    data['top_max'] = data[top_features].max(axis=1)
    data['top_median'] = data[top_features].median(axis=1)
    data['top_sum'] = data[top_features].sum(axis=1)

    return data


def preprocess_data(config):
    logger.info("Preprocessing data...")
    data = load_data(config['data_paths']['raw_data'])
    data = feature_engineer(data)
    y = data['Class']
    X = data.drop('Class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    scaler = StandardScaler()
    X_train['scaled_amount'] = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
    X_test['scaled_amount'] = scaler.transform(X_test['Amount'].values.reshape(-1, 1))
    X_train = X_train.drop('Amount', axis=1)
    X_test = X_test.drop('Amount', axis=1)

    logger.info("Done.")

    return X_train, X_test, y_train, y_test, scaler
