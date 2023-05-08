
import os 
import pandas as pd


def extract_2000_rows():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    read_file = f'{dir_path}/msft_tweets_clean.csv'
    write_file = f'{dir_path}/small_sample_msft_clean.csv'
    if not os.path.isfile(write_file):
        data=pd.read_csv(read_file)

        data = data.sample(n=2000,random_state=2000)

        data.to_csv(write_file)

        print("created file")
    else:
        print("file already exists")

extract_2000_rows()