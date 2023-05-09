from alpha_vantage_keys import alpha_vantage_api_key
import requests
import pandas as pd
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def import_stock_data(save_file=f"{dir_path}/full_msft_data.csv",ticker="msft"):
    if not os.path.isfile(save_file):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={alpha_vantage_api_key}'
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame.from_dict(data)
        df.to_csv(save_file)
        print("created file")
    else:
        print("file already exists")



    

