from Utility.PreProcessing import DataPreProcessing

import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy  as np
import tensorflow as tf
from numpy import load
base_path = ('/home/maira/Data/UASB_HRAP_Modelling')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
data_array_path = (os.path.join(base_path, '/home/maira/github/AmmoniumSensorApproximator/Data'))
model_path = os.path.join(base_path,'Pre-Trained-Models')

# Load Numpy array of inputs and ground Truth

X_train = load(os.path.join(data_array_path,'X_train.npy'))
X_valid = load(os.path.join(data_array_path,'X_valid.npy'))
X_test = load(os.path.join(data_array_path,'X_test.npy'))
nh4_test = load(os.path.join(data_array_path,'nh4_test.npy'))
nh4_train = load(os.path.join(data_array_path,'nh4_train.npy'))
nh4_valid = load(os.path.join(data_array_path,'nh4_valid.npy'))
laggedObservation = 4
nh4_soft_sensor = tf.keras.models.Sequential([
        tf.keras.layers.GRU(100, return_sequences=True, input_shape=[laggedObservation, X_train.shape[2]]),
        tf.keras.layers.GRU(100, return_sequences=True),

        tf.keras.layers.Reshape(target_shape=(20, 20)),

        tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='relu', ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(16, ),

        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(1)
    ])

nh4_soft_sensor.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0010),
        loss=[tf.keras.losses.MeanAbsoluteError()],
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    )

early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

nh4_soft_sensor.fit(
        X_train,
        nh4_train,
        validation_data=(X_valid, nh4_valid),
        batch_size=32,
        epochs=200,
        callbacks=[early_stopping]
    )
trn_y_pred = nh4_soft_sensor.predict(X_train).reshape(-1)
trn_rmse = np.sqrt(mean_squared_error(trn_y_pred, nh4_train))
trn_r2 = r2_score(trn_y_pred, nh4_train)

print("Training Scores")
print("  RMSE: {:7.3f}".format(trn_rmse))
print("  R2:   {:7.3f}".format(trn_r2))
print()

val_y_pred = nh4_soft_sensor.predict(X_valid).reshape(-1)
val_rmse = np.sqrt(mean_squared_error(val_y_pred, nh4_valid))
val_r2 = r2_score(val_y_pred, nh4_valid)

print("Validation Scores")
print("  RMSE: {:7.3f}".format(val_rmse))
print("  R2:   {:7.3f}".format(val_r2))
print()

tst_y_pred = nh4_soft_sensor.predict(X_test).reshape(-1)
nh4_test = nh4_test[:tst_y_pred.shape[0]]

tst_rmse = np.sqrt(mean_squared_error(tst_y_pred, nh4_test))
tst_r2 = r2_score(tst_y_pred, nh4_test)
tst_mae = mean_absolute_error(tst_y_pred, nh4_test)
print("Testing Scores")
print("  RMSE: {:7.3f}".format(tst_rmse))
print("  R2:   {:7.3f}".format(tst_r2))
print()