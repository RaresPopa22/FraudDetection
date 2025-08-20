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

def preprocess_data(config):
    data = load_data(config['data_paths']['raw_data'])
    scaler = StandardScaler()
    data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)

    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    smote = SMOTE(random_state=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled, X_test, y_test

def preprocess_data_tree(config):
    X_train_resampled, y_train_resampled, X_test, y_test = preprocess_data(config)

    X_train_scaled, X_train_eval, y_train, y_train_eval = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.15, random_state=1, stratify=y_train_resampled
    )

    return X_train_scaled, y_train, X_train_eval, y_train_eval, X_test, y_test


