About the data:

DIR: /past_project_clean_data
    File: msft_tweets_clean.csv
        Info: This data contains the processed Twitter data that includes the string "Microsoft" or "msft"
        Rows: 451306
        Cols:
            0: X (each tweet id)
            1: author_id (author's id number)
            2: text (the actual tweet in string format)
            3: created_at (the timestep in YYYY-MM-DDTHH:MM:SS.DDDZ format)
    
    File: msft_tweets_clean.pkl
        Info: Pandas df in a pickle file format of msft_tweets_clean.csv
    
    File: convert_data_into_df.py
        Call load_clean_data_dictionary() to compress file into pickle
        Call load_file_from_pkl() to decrompress file into df


DIR: /stock_price_data
    File: full_msft_data.csv
        Info: This dataset contains the daily msft data from 2000 till 2023
    
    File: av_load_data.py
        Info: Contains functions importing the data and manipulating the data.
        