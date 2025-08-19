import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Starting data preprocessing...")

data = pd.read_csv('../data/creditcard.csv')
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

smote = SMOTE(random_state=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

X_train_reshaped = X_train_resampled.values.reshape(X_train_resampled.shape[0], 1, X_train_resampled.shape[1])
X_test_reshaped = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

print("Saving preprocessed data to preprocessed_data.npz...")
np.savez_compressed(
    '../data/preprocessed/preprocess_data.npz',
    X_train = X_train_reshaped,
    y_train = y_train_resampled,
    X_test = X_test_reshaped,
    y_test = y_test.values
)

print("Preprocessing complete")