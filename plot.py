import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
import argparse
import os
import re

from data_manager.data_manager import DataManager


class Plot:

    DATA_PATH = './data/'
    CSV_EXTENSION = '.csv'
    PREDICTION_FILE_NAME = '_predictions' + CSV_EXTENSION
    TRADES_FILE_NAME = '_trades' + CSV_EXTENSION

    @staticmethod
    def display(symbol=None):
        data_manager = DataManager()
        style.use('ggplot')

        if symbol is None:
            # Get a list of all CSV files in the DATA_PATH directory
            files = [f for f in os.listdir(
                Plot.DATA_PATH) if re.match(r'^[A-Za-z]+\.csv$', f)]
        else:
            files = [symbol + Plot.CSV_EXTENSION]

        # Create a new figure
        plt.figure()

        for file in files:
            # Extract the stock symbol from the filename
            symbol = file.replace(Plot.CSV_EXTENSION, '')

            # Construct the file paths
            data_file = os.path.join(
                Plot.DATA_PATH, symbol + Plot.CSV_EXTENSION)
            prediction_file = os.path.join(
                Plot.DATA_PATH, symbol + Plot.PREDICTION_FILE_NAME)
            trades_file = os.path.join(
                Plot.DATA_PATH, symbol + Plot.TRADES_FILE_NAME)

            # Check if the files exist
            if not os.path.exists(data_file) or not os.path.exists(prediction_file):
                print(f"Files for symbol {symbol} do not exist.")
                return

            # Read the data from the files
            data_df = pd.read_csv(data_file)
            prediction_df = pd.read_csv(prediction_file)

            try:
                trades_df = pd.read_csv(trades_file)
            except FileNotFoundError:
                trades_df = None

            data_df['date'] = pd.to_datetime(data_df['date'])
            data_df.set_index('date', inplace=True)
            data_df = data_manager.add_technical_indicators(data_df)

            prediction_df['date'] = pd.to_datetime(prediction_df['date'])
            prediction_df.set_index('date', inplace=True)

            if trades_df is not None:
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trades_df.set_index('date', inplace=True)

            # Get the first prediction date
            first_prediction_date = prediction_df.index.min().date()
            # Convert the date back to datetime
            first_prediction_date = pd.to_datetime(first_prediction_date)

            # # Filter the data_df to only include rows after the first prediction date
            # data_df = data_df[data_df.index >= first_prediction_date]

            # Create subplots
            fig, axs = plt.subplots(2, figsize=(12, 6), gridspec_kw={
                                    'height_ratios': [3, 2]})

            # Plot the data on the first subplot
            axs[0].plot(data_df.index, data_df['Close'], label='Price')
            axs[0].plot(data_df.index, data_df['EMA_10'],
                        label='Exponential Moving Average (10 days)', color='purple')

            if trades_df is not None:
                colors = trades_df['side'].map({'buy': 'green', 'sell': 'red'})

                axs[0].scatter(trades_df.index, trades_df['price'],
                               label='Buy/Sell', color=colors)
                for i in range(len(trades_df)):
                    text = f"\nSide: {trades_df['side'].iloc[i]}\nQTY: {trades_df['quantity'].iloc[i]}\nPrice: {trades_df['price'].iloc[i]}"

                    axs[0].text(
                        trades_df.index[i], trades_df['price'].iloc[i], text, color=colors.iloc[i])

            axs[0].legend(loc='best')
            axs[0].set_title(f"Price for {symbol}")

            first_group = True

            # Group by week and plot each group separately
            for _, group in prediction_df.groupby(pd.Grouper(freq='W')):
                if first_group:
                    axs[1].plot(group.index, group['prediction'],
                                label='Prediction', color='orange')
                    first_group = False
                axs[1].plot(group.index, group['prediction'], color='orange')

            # Add a horizontal line at prediction threshold
            axs[1].axhline(y=0.7, color='r', linestyle='--')
            axs[1].legend(loc='best')
            axs[1].set_title(f"Predictions for {symbol}")

        # Enable pan and zoom functionality
        plt.rcParams['toolbar'] = 'toolmanager'
        plt.rcParams['axes.grid'] = True

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Plot data from the CSV files.')
    parser.add_argument('--symbol', help='The stock symbol.')
    args = parser.parse_args()

    if args.symbol is None:
        Plot.display()
    else:
        Plot.display(args.symbol)
