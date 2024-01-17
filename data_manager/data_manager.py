'''
    This class will be used to interface with the CSV files that will store the historical stock data
    as well as recording the predictions, buy/sell orders, etc.
'''

# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import pandas_ta as ta
import csv


class DataManager:

    DATA_PATH = './data/'

    def __init__(self):
        pass

    # This function will be used to get the data from the CSV files
    # If the data does not exist, it will be downloaded from the Yfinance api
    # The data will be returned as a list of pandas dataframes

    def get_data(self, watchlist):
        data = {}

        for symbol in watchlist:
            file_path = self.DATA_PATH + symbol + '.csv'

            if not os.path.isfile(file_path):
                # if the file does not exist, download the data from Yfinance api
                self._download_data(symbol)

            # read the data from the csv file
            df = pd.read_csv(file_path, index_col='date')

            data[symbol] = df

        return data
    
    def store_prediction(self, symbol, prediction):
        file_path = self.DATA_PATH + symbol + '_predictions.csv'

        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.isfile(file_path):
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'prediction'])

        # Append the new row to the file
        new_row = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), prediction]
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)

    # Pull data from the yfinance api
    def _download_data(self, symbol):
        ticker = yf.Ticker(symbol)

        # Get the historical data for the ticker
        data = ticker.history(period='2y', interval='1d')

        # Fetch NASDAQ data
        nasdaq = yf.Ticker("^IXIC")
        nasdaq_df = nasdaq.history(period="2y", interval='1d')
        nasdaq_df = nasdaq_df['Close'].rename('NASDAQ_Close')

        # Join the NASDAQ data with the original data
        data = data.join(nasdaq_df)

        data['52_week_high'] = data['Close'].rolling(window=252).max()
        data['52_week_low'] = data['Close'].rolling(window=252).min()
        data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
        

        # Drop todays date since we are going to be appending that ourselves
        data.index = data.index.date
        today = datetime.datetime.now().date()
        data.drop(data[data.index == today].index, inplace=True)

        # Get the last 150 days of data so we can recalculate the indicators.
        data = data.tail(150)

        file_path = self.DATA_PATH + symbol + '.csv'

        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data.index.name = 'date'
        data.to_csv(file_path, index=True)

    
    def _add_technical_indicators(self, dataframe):
        dataframe.ta.sma(length=10, column='Volume', append=True)
        dataframe.rename(columns={'SMA_10': 'avgTradingVolume'}, inplace=True)
        dataframe.ta.sma(length=30, append=True)
        dataframe.ta.ema(length=10, append=True)
        dataframe.ta.macd(append=True)
        dataframe.ta.rsi(length=10, append=True)
        dataframe.ta.bbands(append=True)
        dataframe.ta.adx(length=14, append=True)
        dataframe.ta.ichimoku(append=True)
        dataframe.ta.atr(length=14, append=True)
        dataframe.ta.stoch(high='High', low='Low',
                      close='Close', k=14, d=3, append=True)
        dataframe.ta.psar(high='High', low='Low', close='Close', append=True)


        dataframe['week_day'] = dataframe.index.dayofweek + 1
        dataframe['EMA_10_diff'] = dataframe['EMA_10'].diff()
        dataframe['EMA_10_trend'] = dataframe['EMA_10_diff'].apply(
            lambda x: 1 if x > 0 else 0)
        dataframe['PSAR_combined'] = dataframe['PSARl_0.02_0.2'].fillna(
            dataframe['PSARs_0.02_0.2'])
        

        dataframe.drop(columns=['PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2', 'PSARr_0.02_0.2', 'ICS_26'], inplace=True)
        dataframe.drop(columns=['EMA_10_diff'], inplace=True)
       
        return dataframe

    
    def refresh_dataframe(self, dataframe, new_row):
        index = new_row.pop('date')
        new_row_df = pd.DataFrame(new_row, index=[index])
        
        new_row_df['52_week_high'] = max(
            new_row_df['Close'].iloc[-1], dataframe['52_week_high'].iloc[-1])
        new_row_df['52_week_low'] = min(
            new_row_df['Close'].iloc[-1], dataframe['52_week_low'].iloc[-1])

        refreshed_df = pd.concat([dataframe, new_row_df])
        refreshed_df['Volume'] = refreshed_df['Volume'].ffill()
        refreshed_df['NASDAQ_Close'] = refreshed_df['NASDAQ_Close'].ffill()
        refreshed_df.index = pd.to_datetime(refreshed_df.index)
        refreshed_df = self._add_technical_indicators(refreshed_df).tail(20)

        return refreshed_df
    
    def update_daily_data(self, watchlist):
        for symbol in watchlist:
            self._download_data(symbol)

    def store_trade(self, symbol, side, quantity, price, prediction):
        file_path = self.DATA_PATH + symbol + '_trades.csv'

        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.isfile(file_path):
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'side', 'quantity', 'price', 'prediction'])

        # Append the new row to the file
        new_row = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), side, quantity, price, prediction]
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)

def main():
    # Generate a data manager object
    data_manager = DataManager()

    # Get the data
    data = data_manager.get_data(['AAPL', 'TSLA'])

    # Print the data
    print(data)


if __name__ == '__main__':
    main()
