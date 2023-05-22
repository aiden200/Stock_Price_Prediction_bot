import logging as log
from stock_price_data.av_load_data import *
import numpy as np

log.basicConfig(filename='data.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
log.getLogger().setLevel(log.INFO)
log.getLogger().setLevel(log.WARNING)


def transform_price_data(log, start_date, end_date, read_file="stock_price_data/full_msft_price_data.csv", write_file="clean_price_msft.csv"):
    log.info("Function transform_price_data: Starting transforming price data")
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


def append_data(log, output_file = "full_msft_quantatative_metrics.csv", metrics=["EMA", "MOM", "SMA"]):
    try:
        log.info("Function append_data: Starting appending process of the data")
        price_data = pd.read_csv("stock_price_data/full_msft_price_data.csv")
        price_data = price_data.iloc[5:]
        all_data = []
        for metric in metrics:
            price_data.loc[:, metric] = [0]*len(price_data)
            df = pd.read_csv(f"stock_price_data/full_msft_{metric}_data.csv")
            df = df.iloc[7:]
            all_data.append(df)
        missing_count = 0
        for i in range(len(price_data)):
            date = price_data.iloc[i, 0]
            for j in range(len(metrics)):
                curr_df = all_data[j]
                data = curr_df.loc[curr_df['Unnamed: 0'] == date]
                if len(data) == 0:
                    price_data.iloc[i, 3+j] = 0
                    missing_count += 1
                else:
                    value = data.iloc[0,2]
                    value = float(value.split(":")[1].replace(" ","")[1:-2])
                    price_data.iloc[i, 3+j] = value
        price_data.to_csv(output_file)
        log.info(f"Missing Count: {missing_count}")
        print(f"Missing Count: {missing_count}")
    except Exception as e:
        log.warning(f"Failed in function append_data with error {e}")
        print(f"Failed in function append_data with error {e}")


append_data(log)

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

# transform_data(log)

