import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import logging as log
import os
from pathlib import Path


def load_mnist_data():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    return x_train, x_test, y_train, y_test

dir_path = os.path.dirname(os.path.realpath(__file__))
path = Path(dir_path)
parent_path = path.parent.absolute()


log.basicConfig(filename='log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
log.getLogger().setLevel(log.INFO)
log.getLogger().setLevel(log.WARNING)


def load_price_data(log=None):
    if log:
        log.info("Loading raw price data")
    try:
        df = pd.read_csv(f"{parent_path}/data/clean_price_msft.csv")
        raw_price_data = df.to_numpy()
        return raw_price_data
    except Exception as e:
        if log:
            log.warning(f"Error in function load_price_data with error: {e}")
        return None

def one_layer_lstm_model(log=None):
    if log:
        log.info("Loading LSTM model")
    model = keras.Sequential()
    model.add(layers.LSTM(64, input_shape=(None, 28)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10))
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="sgd",
        metrics=["accuracy"],
    )	

    if log:
        log.info(f"Loaded one layer LSTM model")
    return model


def train_model():
    log.info("Starting training log")
    # x_train, x_test, y_train, y_test = load_price_data(log)
    x_train, x_test, y_train, y_test = load_mnist_data()
    model = one_layer_lstm_model(log)
    model.fit(
        x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=10
    )

train_model()

