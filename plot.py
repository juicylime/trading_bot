import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
import argparse
import os

class Plot:

    DATA_PATH = './data/'
    CSV_EXTENSION = '.csv'
    PREDICTION_FILE_NAME = '_predictions' + CSV_EXTENSION
    TRADES_FILE_NAME = '_trades' + CSV_EXTENSION

    @staticmethod
    def display(symbol='AMZN'):
        # Construct the file paths
        data_file = os.path.join(Plot.DATA_PATH, symbol + Plot.CSV_EXTENSION)
        prediction_file = os.path.join(Plot.DATA_PATH, symbol + Plot.PREDICTION_FILE_NAME)
        # trades_file = os.path.join(Plot.DATA_PATH, symbol + Plot.TRADES_FILE_NAME)

        # Check if the files exist
        # if not os.path.exists(data_file) or not os.path.exists(prediction_file) or not os.path.exists(trades_file):
        if not os.path.exists(data_file) or not os.path.exists(prediction_file):
            print(f"Files for symbol {symbol} do not exist.")
            return
        

        # Read the data from the files
        data_df = pd.read_csv(data_file)
        prediction_df = pd.read_csv(prediction_file)
        # trades_df = pd.read_csv(trades_file)

        # Convert the date in the data dataframe to datetime and set it as index
        data_df['date'] = pd.to_datetime(data_df['date'])
        data_df.set_index('date', inplace=True)

        # Convert the date-time in the prediction dataframe to datetime
        prediction_df['date'] = pd.to_datetime(prediction_df['date'])
        prediction_df.set_index('date', inplace=True)

        # Get the first prediction date
        first_prediction_date = prediction_df.index.min().date()
        # Convert the date back to datetime
        first_prediction_date = pd.to_datetime(first_prediction_date)

        # Filter the data_df to only include rows after the first prediction date
        data_df = data_df[data_df.index >= first_prediction_date]


        # Create subplots
        fig, axs = plt.subplots(2, figsize=(12, 12))

        # Plot the data on the first subplot
        axs[0].plot(data_df.index, data_df['Close'], label='Data')
        axs[0].legend(loc='best')
        axs[0].set_title(f"Data for {symbol}")

        # Group by week and plot each group separately
        for _, group in prediction_df.groupby(pd.Grouper(freq='W')):
            axs[1].plot(group.index, group['prediction'], label='Predictions', color='orange')

        # Add a horizontal line at prediction threshold
        axs[1].axhline(y=0.7, color='r', linestyle='--')
        axs[1].legend(loc='best')
        axs[1].set_title(f"Predictions for {symbol}")

        # Set some basic styling
        plt.style.use('ggplot')

        # Enable pan and zoom functionality
        plt.rcParams['toolbar'] = 'toolmanager'
        plt.rcParams['axes.grid'] = True

        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot data from the CSV files.')
    parser.add_argument('--symbol', help='The stock symbol.')
    args = parser.parse_args()

    if args.symbol is None:
        Plot.display()
    else:
        Plot.display(args.symbol)
