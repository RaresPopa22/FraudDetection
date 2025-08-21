import numpy as np
import pandas as pd
import yaml

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

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
    data['Hour'] = data['Time'].apply(lambda x: np.floor(x / 3600))
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 23.0)
    data['Hour_cos'] = np.sin(2 * np.pi * data['Hour'] / 23.0)
    data.drop('Hour', axis=1, inplace=True)
    data.drop('Time', axis=1, inplace=True)

    return data

def preprocess_data(config):
    data = load_data(config['data_paths']['raw_data'])
    scaler = StandardScaler()
    data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data = feature_engineer(data)
    data = data.drop('Amount', axis=1)

    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    return X_train, X_test, y_train, y_test

def apply_SMOTE(X_train, y_train):
    smote = SMOTE(random_state=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled


