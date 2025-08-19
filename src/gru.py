import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Bidirectional
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, auc
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, GRU
from tensorflow.keras.models import Model
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"GPU device(s) found: {gpu_devices}")
else:
    print("No GPU devices found. TensorFlow is running on the CPU.")



BATCH_SIZE = 16384
NUM_CORES = 12
tf.config.threading.set_inter_op_parallelism_threads(NUM_CORES)
tf.config.threading.set_intra_op_parallelism_threads(NUM_CORES)


print("Loading preprocessed data...")
data = np.load("../data/preprocessed/preprocess_data.npz")
X_train_reshaped = data['X_train']
y_train_resampled = data['y_train']
X_test_reshaped = data['X_test']
y_test = data['y_test']
print("Data loaded successfully")

def build_model(hp):
    lstm_units_1 = hp.Int('lstm_units_1', min_value=64, max_value=256, step=64)
    lstm_units_2 = hp.Int('lstm_units_2', min_value=32, max_value=128, step=32)

    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    input_layer = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
    x = GRU(units=lstm_units_1, return_sequences=True)(input_layer)
    x = Dropout(dropout_rate)(x)
    x = GRU(units=lstm_units_2)(x)
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

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, y_train_resampled))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size=len(X_train_reshaped))
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, y_train_resampled))
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.cache()
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(
    train_dataset,
    validation_data = val_dataset,
    epochs=50,
    callbacks=[early_stopping]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.get_best_models(num_models=1)[0]

model.save('../models/best_fraud_detection_model.keras')
print("Best model has been saved successfully")

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