import logging as log
from stock_price_data.av_load_data import *
import numpy as np

log.basicConfig(filename='data.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
log.getLogger().setLevel(log.INFO)
log.getLogger().setLevel(log.WARNING)


def transform_price_data(log, start_date, end_date, read_file="stock_price_data/full_msft_price_data.csv", write_file="clean_price_msft.csv"):
    log.info("Starting transforming price data")
    df = pd.read_csv(read_file)
    df = df.iloc[5:]
    df = df.to_numpy()
    start_index = np.where(df==end_date)[0][0]
    end_index = np.where(df==start_date)[0][0]
    data = []
    for i in range(start_index, end_index + 1):
        # each row appended as start date, open, high, low, close, adjusted close, volume
        row = df[i]
        values = row[2].split("'")
        row_i = [row[0], float(values[3]), float(values[7]), float(values[11]), float(values[15]), float(values[19]), float(values[23])]
        data.append(row_i)
    df = pd.DataFrame.from_dict(data)
    df.to_csv(write_file)
    log.info(f"created {write_file}")

def transform_data(log):

    start_date = "2022-01-03"
    end_date = '2022-12-30'
    transform_price_data(log, start_date, end_date)


def load_all_data():
    log.info("Starting data collection")
    print("Loading all the data, please check data.log for details")

    log.info("Loading Stock metrics data")
    import_stock_data_metrics(log=log)
    import_stock_ema_and_sma_metrics(log=log)
    log.info("Stock metrics data created")

    log.info("Loading article data")
    import_article_data(log)
    log.info("Article data created")
    
    log.info("starting transforming data")
    transform_data(log)
    log.info("transformed data")

transform_data(log)

