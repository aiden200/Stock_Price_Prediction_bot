from .alpha_vantage_keys import alpha_vantage_api_key
import requests
import pandas as pd
import os
import time
dir_path = os.path.dirname(os.path.realpath(__file__))

'''
API limitations: 
    5 API requests per minute and 500 requests per day.

'''
DATA_LIMIT_MESSAGE = {'Note': 'Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day. Please visit https://www.alphavantage.co/premium/ if you would like to target a higher API call frequency.'}

def import_stock_data_metrics(save_file=f"{dir_path}/full_msft_price_data.csv",ticker="msft", log=None):
    try:
        if not os.path.isfile(save_file):
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={alpha_vantage_api_key}'
            r = requests.get(url)
            data = r.json()
            df = pd.DataFrame.from_dict(data)
            df.to_csv(save_file)
            print("created file")
            if log:
                log.info(f"Created file: {save_file}")
        else:
            print("file already exists")
            if log:
                log.warning(f"Function error: import_stock_data_metrics\nfile already exists: {save_file}")
    except Exception as e:
        print(f"import_stock_data_metrics: FAILED with error: {e}")
        log.warning(f"import_stock_data_metrics: FAILED with error: {e}")

def import_stock_ema_and_sma_metrics(save_file=f"{dir_path}/full_msft_price_data.csv",ticker="msft",log=None, metrics = ['EMA','SMA','MOM','VWAP']):
    try:
        for metric in metrics:
            save_file = f"{dir_path}/full_msft_{metric}_data.csv"
            interval = "daily"
            if metric == "VWAP":
                interval = "60min"
            if not os.path.isfile(save_file):
                url = f'https://www.alphavantage.co/query?function={metric}&symbol={ticker}&interval={interval}&time_period=21&series_type=open&apikey={alpha_vantage_api_key}'
                r = requests.get(url)
                data = r.json()
                df = pd.DataFrame.from_dict(data)
                df.to_csv(save_file)
                print("created file")
                if log:
                    log.info(f"Created File {save_file}")
            else:
                print("file already exists")
                if log:
                    log.warning(f"Function error: import_stock_ema_and_sma_metrics\nfile already exists: {save_file}")
    except Exception as e:
        print(f"import_stock_ema_and_sma_metrics: FAILED with error: {e}")
        if log:
            log.warning(f"import_stock_ema_and_sma_metrics: FAILED with error: {e}")




def import_article_data(ticker="msft", log=None):
    '''
    We can only import 50 matching results therefore will start the time at 2021 since thats when we start our predictions
    '''
    start_time = "20220301T0101"
    year = "2022"
    i = 0
    data = ''

    try:
        while year != "2023":
            save_file=f"{dir_path}/article_data/2022/msft_article_data_{i}.csv"
            if not os.path.isfile(save_file):
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={start_time}&limit=200&sort=EARLIEST&apikey={alpha_vantage_api_key}"
                r = requests.get(url)
                data = r.json()
                if data != DATA_LIMIT_MESSAGE:
                    if log:
                        log.info("Hit API limit, wait 1 minute to proceed")
                    df = pd.DataFrame.from_dict(data)
                    df.to_csv(save_file)
                    print("created file")
                    if log:
                        log.info(f"Created file: {save_file}")
                else:
                    print("Hit API limit, waiting 1 minute to proceed")
                    time.sleep(60)
            else:
                print("file already exists")
                if log:
                    log.info(f"import_article_data\nfile already exists: {save_file}")
            if data != DATA_LIMIT_MESSAGE:
                feed = pd.read_csv(save_file).iloc[-1]["feed"]
                index = feed.index('time_published')
                date = feed[index+18:index+26]
                start_time = f'{date}T0101'
                year = start_time[:4]
                i+=1
    except Exception as e:
        print(f"import_article_data: FAILED with error: {e}")
        print(f"start time: {start_time}")
        if log:
            log.warning(f"import_article_data: FAILED with error: {e}")



import_article_data()

