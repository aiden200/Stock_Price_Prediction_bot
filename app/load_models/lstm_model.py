import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import logging as log
import os
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

T = 10
P = 2


dir_path = os.path.dirname(os.path.realpath(__file__))
path = Path(dir_path)
parent_path = path.parent.absolute()


log.basicConfig(filename='log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
log.getLogger().setLevel(log.INFO)
log.getLogger().setLevel(log.WARNING)

def load_mnist_data():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    return x_train, x_test, y_train, y_test

def load_price_data(t = T, p = P, log=None):
    if log:
        log.info("Loading raw price data")
    try:
        df = pd.read_csv(f"{parent_path}/data/clean_price_msft.csv")
        raw_price_data = df.to_numpy()
        #columns are start date, open, high, low, close, adjusted close, volume
        raw_price_data = raw_price_data[:,2:]
        raw_price_data = raw_price_data.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        raw_price_data = scaler.fit_transform(raw_price_data)

        test_seg = raw_price_data[-50:]
        train_seg = raw_price_data[:-51]
        warning_msg = f"Wrong length, train_seg expected 200, got {len(train_seg)}, test_seg expected 50, got {len(test_seg)}"
        assert len(train_seg) == 200 and len(test_seg) == 50, warning_msg
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        index = 0
        while True:
            if (index + t + p) > len(train_seg):
                break
            x_train.append(train_seg[index:index+t])
            y_train.append(train_seg[index+t:index+t+p][:,2])
            index += p
        index = 0
        while True:
            if (index + t + p) > len(test_seg):
                break
            x_train.append(test_seg[index:index+t])
            y_train.append(test_seg[index+t:index+t+p][:,2])
            index += p
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        log.info(f"Produced data. x_train shape: {x_train.shape}")
        print(f"Produced data. x_train shape: {x_train.shape}")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        if log:
            log.warning(f"Error in function load_price_data with error: {e}")
        return None

def plot_price_data(log=None):
    price_data = load_price_data()
    plt.plot(range(len(price_data)),price_data[:, 2], label="Open Value")
    plt.plot(range(len(price_data)),price_data[:, 3], label="Close Value")
    plt.show()



def one_layer_lstm_model(optimizer = "rmsprop", log=None):
    if log:
        log.info("Loading LSTM model")
    model = keras.Sequential()
    model.add(layers.LSTM(64, input_shape=(T, 6))) # 6 is each row size
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(P))
    model.compile(
        loss="mae",
        optimizer=optimizer,
        metrics=["accuracy"],
    )	

    if log:
        log.info(f"Loaded one layer LSTM model")
    return model


def train_model_1():
    log.info("Starting training log")
    x_train, x_test, y_train, y_test = load_price_data(log=log)
    # x_train, x_test, y_train, y_test = load_mnist_data()
    # optimizer = "rmsprop"
    optimizer = "adam"
    model = one_layer_lstm_model(optimizer, log)
    history = model.fit(
        x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=10
    )
    # print(history.history)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.legend()
    plt.show()

train_model_1()

