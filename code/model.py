from pathlib import Path

import tensorflow as tf
from tensorflow import keras


def create_model(X):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=32, input_shape=(X.shape[1], X.shape[2])))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )

    return model


def save_model(save_dir: Path, model):
    if save_dir.exists():
        model.save(save_dir / "model.h5")
    else:
        save_dir.mkdir(exist_ok=True)
        model.save(save_dir / "model.h5")


def load_model(path: Path):
    return tf.keras.models.load_model(path)