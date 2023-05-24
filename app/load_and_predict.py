'''
Full workflow of loading the models and predicting

'''
import logging as log
from load_models.lstm_model import save_best_model
from data.load_all_data import load_all_data
import os
from tensorflow.keras.models import load_model

log.basicConfig(filename='log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
log.getLogger().setLevel(log.INFO)
log.getLogger().setLevel(log.WARNING)



'''
We should only call this function when we need to load new data and retrain our model from scratch. 
Loads the MSFT stock data and saves it in a format that our functions can process.
currently all stock data 2019(start)-2023(start)
Saves the best LSTM model trying different combinations.
TODO: Need to load the BERT model
'''
def load_models_and_data(log):
    try:
        log.info("Loading data and models: This might take a bit of time")
        # Retrieves API data and saves all the quantatative metrics in a file
        load_all_data(log)
        # Finds the best LSTM model
        save_best_model(log)
    except Exception as e:
        log.warning(f"Failed to load models and data with exception {e}")

'''
We can see the best weights and configurations for the LSTM model. Last updated:
05/24/2023
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 160)               108800    
                                                                 
 dropout (Dropout)           (None, 160)               0         
                                                                 
 dense (Dense)               (None, 336)               54096     
                                                                 
 dropout_1 (Dropout)         (None, 336)               0         
                                                                 
 dense_1 (Dense)             (None, 9)                 3033      
                                                                 
=================================================================
Total params: 165,929
Trainable params: 165,929
Non-trainable params: 0
_________________________________________________________________

'''
def display_best_model(log):
    if os.path.exists("load_models/best_model_multistep_sentiment.h5"):
        model = load_model("load_models/best_model_multistep_sentiment.h5")
        model.summary()
    else:
        log.warning("Model Does not exist, run save_best_model() to load LSTM model.")

def predict(log):
    log.info("")

display_best_model(log)