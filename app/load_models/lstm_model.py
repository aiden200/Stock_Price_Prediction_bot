import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential, load_model
from keras.layers import LSTM, RepeatVector, Dropout, Dense, TimeDistributed
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
import math

T = 10
P = 30
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



def load_price_data(t = T, p = P, log=None, extended=True):
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
        test_n = math.floor(len(raw_price_data) * .2)
        test_seg = raw_price_data[-test_n:]
        train_seg = raw_price_data[:-test_n-1]
        # warning_msg = f"Wrong length, train_seg expected 200, got {len(train_seg)}, test_seg expected 50, got {len(test_seg)}"
        # assert len(train_seg) == 200 and len(test_seg) == 50, warning_msg
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
            if extended:
                y_train.append(train_seg[index+t:index+t+p])
            else:
                y_train.append(train_seg[index+t:index+t+p][:,2])

            index += p
        index = 0
        while True:
            if (index + t + p) > len(test_seg):
                break
            # x_test.append(x_scaler.fit_transform(test_seg[index:index+t]))
            # y_test.append(y_scaler.fit_transform(test_seg[index+t:index+t+p][:,2]))
            x_test.append(test_seg[index:index+t])
            if extended:
                y_test.append(test_seg[index+t:index+t+p])
            else:
                y_test.append(test_seg[index+t:index+t+p][:,2])
            index += p
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        if log:
            log.info(f"Produced data. x_train shape: {x_train.shape}")
        print(f"Produced data. x_train shape: {x_train.shape}")
        print(f"Produced data. y_train shape: {y_train.shape}")
        print(f"Produced data. x_test shape: {x_test.shape}")
        print(f"Produced data. y_test shape: {y_test.shape}")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        print(e)
        print("Cannot load data")
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
    model.add(LSTM(units=50,return_sequences=True,input_shape=(T, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True,input_shape=(T, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True,input_shape=(T, n_features)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=["accuracy", "mse"])
    if log:
        log.info(f"Loaded four layer LSTM model")
    return model


def train_model_1():
    log.info("Starting training log")
    x_train, x_test, y_train, y_test = load_price_data(log=log, extended=True)
    # x_train, x_test, y_train, y_test = load_mnist_data()
    # optimizer = "rmsprop"
    optimizer = "adam"
    model = four_layer_lstm_model(optimizer, log)
    checkpoint_path = "best_model.hdf5"
    checkpoint = [
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                        monitor="val_loss",
                                        save_best_only = True,
                                        mode='min') 
    ]
    earlystopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=0)
    callbacks = [checkpoint, earlystopping]

    # Training the model
    epochs = 500
    batch_size = 32
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, 
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks = callbacks,
                        verbose = 1)
    
    plt.figure(figsize=(16,7))
    plt.title('MSFT Stock Price Prediction')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    
    
    model_from_saved_checkpoint = load_model(checkpoint_path)
    y_hat= model_from_saved_checkpoint.predict(x_test)

    

    
    y_hat_open = np.array([scaler.inverse_transform(item)[:,0] for item in y_hat])
    y_true_open = np.array([scaler.inverse_transform(item)[:,0] for item in y_test])
    print(y_hat.shape)
    print(y_hat_open.shape)
    print(y_test.shape)
    print(y_true_open.shape)

    plt.plot(y_true_open[:,0], label='True')
    plt.plot(y_hat_open[:,0], label='LSTM')
    plt.title("LSTM's_Prediction")
    plt.xlabel('Time steps')
    plt.ylabel('Prices')
    plt.legend()
    plt.show()



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
    

    model.add(RepeatVector(P))
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
    
    model.add(LSTM(units=hp.Int("n_units2",
                               min_value=32,
                               max_value=256,
                               step=16),
                  return_sequences = True))

    model.add(TimeDistributed(Dense(n_features)))
    # model.add(Dense(n_features))

    model.compile(optimizer = "adam",
                 loss = "mean_absolute_error",
                 metrics=["accuracy", "mse"]
                 )
    
    return model

def save_best_model(log, best_model_path):
    log.info("Starting training log")
    x_train, x_test, y_train, y_test = load_price_data(log=log)

    tuner = RandomSearch(build_model,
                     objective="val_loss",
                     max_trials=5,
                     seed=9999,
                     executions_per_trial=20, # try 5, 10, 20
                     directory="",
                     project_name="RandomSearch_Multistep2")

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
    best_model.save(best_model_path)

    # Evaluate the best model with test data
    loss = best_model.evaluate(x_test, y_test)
    print(loss)



def train_weights_on_best_model(checkpoint_path, model_path=f"{dir_path}/best_model_multistep2.h5", log=None):
    x_train, x_test, y_train, y_test = load_price_data(log=log, extended=True)
    model = load_model(model_path)
    checkpoint = [
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                        monitor="val_loss",
                                        save_best_only = True,
                                        mode='min') 
    ]
    earlystopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=0)
    callbacks = [checkpoint, earlystopping]

    # Training the model
    epochs = 500
    batch_size = 32
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, 
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks = callbacks,
                        verbose = 1)
    
    return f"{dir_path}/{checkpoint_path}"

def predict_using_weighted_model(x, checkpoint_path, y=None, log=None):
    model_from_saved_checkpoint = load_model(checkpoint_path)
    y_hat= model_from_saved_checkpoint.predict(x)
    
    y_hat_open = np.array(scaler.inverse_transform(y_hat[0]))[:,0]
    y_true_open = np.array(scaler.inverse_transform(y[0]))[:,0]
    # y_hat_open = np.array([item[:,0] for item in y_hat])
    # y_true_open = np.array([item[:,0] for item in y])
    # y_hat_open = np.array([scaler.inverse_transform(item)[:,0] for item in y_hat])
    # y_true_open = np.array([scaler.inverse_transform(item)[:,0] for item in y])
    print(y_hat.shape)
    print(y_hat_open.shape)
    print(y.shape)
    print(y_true_open.shape)

    plt.plot(y_true_open, label='True')
    plt.plot(y_hat_open, label='LSTM')
    plt.title("LSTM's_Prediction")
    plt.xlabel('Time steps')
    plt.ylabel('Prices')
    plt.legend()
    plt.show()

if __name__=="__main__":
    best_model_path = "best_model_multistep2.h5"
    if not os.path.exists(best_model_path):
        print("Loading new LSTM model with hp")
        save_best_model(log=log, best_model_path=best_model_path)
    checkpoint_path = "best_model2.hdf5"
    if not os.path.exists(checkpoint_path):
        print("Loading new checkpoint model")
        checkpoint_path = train_weights_on_best_model(checkpoint_path=checkpoint_path, model_path = best_model_path, log = log)
    _, x, _, y = load_price_data(log=log, extended=True)
    predict_using_weighted_model(x=x,y=y, checkpoint_path = checkpoint_path)

