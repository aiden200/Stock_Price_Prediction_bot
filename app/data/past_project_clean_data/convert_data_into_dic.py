'''
Converts msft_tweets_clean.csv data into a panda df
'''
import pickle
import pandas as pd
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


def load_clean_data_dictionary(log=None):
    save_file = f'{dir_path}/msft_tweets_clean.pkl'
    read_file = f'{dir_path}/msft_tweets_clean.csv'

    data=pd.read_csv(read_file)

    try:
        data.to_pickle(save_file) 
        print(f'dictionary saved successfully to file: {save_file}')
        if log:
            log.info(f"dictionary saved successfully to file: {save_file}")
    except Exception as e:
        print(f"Failed to load file msft_tweets_clean.csv into dictionary with error: {e} ")
        if log:
            log.error(f"Failed to load file msft_tweets_clean.csv into dictionary with error: {e} ")


def load_file_from_pkl(file_name=f'{dir_path}/msft_tweets_clean.pkl', log=None):
    try:
        df = pd.read_pickle(file_name)  
        print(f'dictionary successfully loaded from file: {file_name}')
        if log:
            log.info(f'dictionary successfully loaded from file: {file_name}')
        return df
    
    except Exception as e:
        print(f"Failed to load dictionary from file: {file_name} with exception: {e} ")
        if log:
            log.error(f"Failed to load dictionary from file: {file_name} with exception: {e} ")


if __name__ == "__main__":
   load_clean_data_dictionary()