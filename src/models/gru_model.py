import tensorflow as tf
import keras_tuner as kt

from tensorflow.keras.layers import Dense, Input, Dropout, GRU
from tensorflow.keras.models import Model
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from keras.src.callbacks import EarlyStopping

class GRUHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        lstm_units_1 = hp.Int('lstm_units_1', min_value=64, max_value=256, step=64)
        lstm_units_2 = hp.Int('lstm_units_2', min_value=32, max_value=128, step=32)

        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        input_layer = Input(shape=self.input_shape)
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

def tune_gru(config, X_train_reshaped, y_train_resampled):
    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
    hypermodel = GRUHyperModel(input_shape=input_shape)

    tuner = kt.Hyperband(
        hypermodel,
        objective=kt.Objective('val_recall', direction='max'),
        max_epochs=20,
        factor=3,
        directory='keras_tuner_dir',
        project_name='fraud_detection'
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, y_train_resampled))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train_reshaped))
    train_dataset = train_dataset.batch(config['batch_size'])
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, y_train_resampled))
    val_dataset = val_dataset.batch(config['batch_size'])
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[early_stopping]
    )

    return tuner.get_best_models(num_models=1)[0]

