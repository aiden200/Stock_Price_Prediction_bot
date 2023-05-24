import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_tuner import HyperParameters
from keras_tuner.tuners import RandomSearch
import pandas as pd
import numpy as np
import logging as log
import os
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

T = 10
P = 1
n_features = 9


dir_path = os.path.dirname(os.path.realpath(__file__))
path = Path(dir_path)
parent_path = path.parent.absolute()


log.basicConfig(filename='log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
log.getLogger().setLevel(log.INFO)
log.getLogger().setLevel(log.WARNING)
# x_scaler = MinMaxScaler(feature_range=(0, 1))
# y_scaler = MinMaxScaler(feature_range=(0, 1))
scaler = MinMaxScaler(feature_range=(0, 1))


def load_mnist_data():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    return x_train, x_test, y_train, y_test



def load_price_data(t = T, p = P, log=None):
    if log:
        log.info("Loading raw price data")
    try:
        '''
        TODO: we can also generate data using keras.utils.timeseries_dataset_from_array
        '''
        df = pd.read_csv(f"{parent_path}/data/clean_price_msft.csv")
        raw_price_data = df.to_numpy()
        #columns are start date, open, high, low, close, adjusted close, volume
        raw_price_data = raw_price_data[:,2:]
        raw_price_data = raw_price_data.astype('float32')
        raw_price_data = scaler.fit_transform(raw_price_data)

        # # normalize features

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

            # x_train.append(x_scaler.fit_transform(train_seg[index:index+t]))
            # y_train.append(y_scaler.fit_transform(train_seg[index+t:index+t+p][:,2]))
            x_train.append(train_seg[index:index+t])
            y_train.append(train_seg[index+t:index+t+p][:,2])
            index += p
        index = 0
        while True:
            if (index + t + p) > len(test_seg):
                break
            # x_test.append(x_scaler.fit_transform(test_seg[index:index+t]))
            # y_test.append(y_scaler.fit_transform(test_seg[index+t:index+t+p][:,2]))
            x_test.append(test_seg[index:index+t])
            y_test.append(test_seg[index+t:index+t+p][:,2])
            index += p
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        log.info(f"Produced data. x_train shape: {x_train.shape}")
        print(f"Produced data. x_train shape: {x_train.shape}")
        print(f"Produced data. x_test shape: {x_test.shape}")
        print(f"Produced data. y_test shape: {y_test.shape}")
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
    model.add(layers.LSTM(64, input_shape=(T, n_features))) # 6 is each row size
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(P))
    # we use a callback to save the best performing model
    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["accuracy", "mae"]
    )	

    if log:
        log.info(f"Loaded one layer LSTM model")
    return model

def four_layer_lstm_model(optimizer="adam", log=None):
    model = keras.Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(T, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=P))
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=["accuracy", "mse"])
    if log:
        log.info(f"Loaded four layer LSTM model")
    return model


def train_model_1():
    log.info("Starting training log")
    x_train, x_test, y_train, y_test = load_price_data(log=log)
    # x_train, x_test, y_train, y_test = load_mnist_data()
    # optimizer = "rmsprop"
    optimizer = "adam"
    model = four_layer_lstm_model(optimizer, log)
    callbacks = [
        keras.callbacks.ModelCheckpoint("one_layer_dense.keras",
                                        save_best_only = True,
                                        monitor='loss') 
    ]
    history = model.fit(
        x_train, 
        y_train, 
        validation_data=(x_test, y_test), 
        batch_size=64, 
        epochs=10,
        callbacks = callbacks
    )

    # model = keras.models.load_model("one_layer_dense.keras")
    # print(f"Test MAE: {model.evaluate(x_test, y_test)}")

    # print(history.history)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.legend()
    plt.show()

    yhat = model.predict(x_test)
    # print(x_test.shape, yhat.shape)
    # y_train = scaler.inverse_transform(y_train)
    # yhat = scaler.inverse_transform(yhat)
    plt.title('MSFT Stock Price Prediction')
    plt.plot(range(len(y_test)), y_test, label="actual test")
    plt.plot(range(len(yhat)), yhat, label="predicted test")
    plt.legend()
    plt.show()
    # yhat2 = model.predict(x_test)
    # plt.plot(range(len(y_test)), y_test, label="actual test")
    # plt.plot(range(len(yhat2)), yhat2, label="predicted test")

# train_model_1()

# hp stands for hyperparameters
def build_model(hp):
    model = Sequential()
    
    # number of dense layer
    dense = hp.Int("n_dense",
                  min_value=0,
                  max_value=3)

    # first LSTM-layer
    model.add(LSTM(units=hp.Int("n_units1",
                               min_value=32,
                               max_value=512,
                               step=16
                               ),
                  input_shape=(T, n_features)))
    

    # model.add(RepeatVector(n_steps_out))
    if dense > 0:
        for layer in range(dense):
            model.add(Dropout(hp.Float("v_dropout_dense" + str(layer + 1),
                                        min_value = 0.05,
                                        max_value = 0.2,
                                        step = 0.05)))
            model.add(Dense(units=hp.Int("n_units_dense" + str(layer + 1),
                                        min_value = 32,
                                        max_value = 512,
                                        step = 16),
                            activation=hp.Choice("v_activation_dense",
                                                values=["relu", "tanh", "sigmoid"],
                                                default="relu")))
    
    model.add(Dropout(hp.Float("v_dropout",
                              min_value=0.05,
                              max_value=0.2,
                              step=0.05)))
    
    # model.add(LSTM(units=hp.Int("n_units2",
    #                            min_value=32,
    #                            max_value=256,
    #                            step=16),
    #               return_sequences = True))

    # model.add(TimeDistributed(Dense(n_features)))
    model.add(Dense(n_features))

    model.compile(optimizer = "adam",
                 loss = "mean_absolute_error",
                 metrics=["accuracy", "mse"]
                 )
    
    return model

def save_best_model():
    log.info("Starting training log")
    x_train, x_test, y_train, y_test = load_price_data(log=log)

    tuner = RandomSearch(build_model,
                     objective="val_loss",
                     max_trials=3,
                     seed=9999,
                     executions_per_trial=5, # try 5, 10, 20
                     directory="",
                     project_name="RandomSearch_Multistep")

    early_stopping_cb = EarlyStopping(patience=15, restore_best_weights=True)

    tuner.search(x_train, 
                y_train, 
                epochs=500, 
                batch_size=16, 
                validation_split=0.25,
                callbacks=[early_stopping_cb],
                verbose=1)

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # save best model for that variable combination
    best_model.save("best_model_multistep_sentiment.h5")

    # Evaluate the best model with test data
    loss = best_model.evaluate(x_test, y_test)
    print(loss)

save_best_model()

