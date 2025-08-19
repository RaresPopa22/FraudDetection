import keras_tuner as kt
import pandas as pd
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Bidirectional
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.models import Model
from tensorflow.python.keras.metrics import Precision, Recall
from imblearn.over_sampling import SMOTE
from tensorflow.keras.optimizers import Adam

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

print(f'X_train_resampled.shape={X_train_resampled.shape}')
X_train_reshaped = X_train_resampled.values.reshape(X_train_resampled.shape[0], 1, X_train_resampled.shape[1])
X_test_reshaped = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

def build_model(hp):
    lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=128, step=32)
    lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=64, step=16)

    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    input_layer = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
    x = Bidirectional(LSTM(units=lstm_units_1, return_sequences=True))(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units=lstm_units_2))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=16, activation='relu')(x)
    output_layer = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[Precision(name='precision'), Recall(name='recall')])

    return model

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective('val_recall', direction='max'),
    max_epochs=20,
    factor=3,
    directory='keras_tuner_dir',
    project_name='fraud_detection'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(
    X_train_reshaped,
    y_train_resampled,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.get_best_models(num_models=1)[0]

y_pred_prob = model.predict(X_test_reshaped)
y_pred = (y_pred_prob > 0.5).astype(int)

report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
print("Classification Report:")
print(report)

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
auprc = auc(recall, precision)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUPRC={auprc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Fraud Detection')
plt.legend(loc='best')
plt.grid(True)
plt.show()