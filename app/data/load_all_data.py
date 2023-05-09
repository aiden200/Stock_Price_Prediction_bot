import logging as log
from stock_price_data.av_load_data import *

log.basicConfig(filename='data.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
log.getLogger().setLevel(log.INFO)
log.getLogger().setLevel(log.WARNING)


def load_all_data():
    log.info("Starting data collection")
    print("Loading all the data, please check data.log for details")

    log.info("Loading Stock metrics data")
    import_stock_data_metrics(log=log)
    import_stock_ema_and_sma_metrics(log=log)
    log.info("Stock metrics data created")

    log.info("Loading article data")
    



