''' 
    This file will serve as the main driver for the application
    The application is a trading bot that will use the Alpaca API to trade stocks
    The bot will use a trained model to predict the value of the 5 day moving average 3 days into the future

    What the bot will do
    
    Prepare the data:
    1. Authenticate with Alpaca API
    2. Get the watchlist of stocks
    3. Checks to see if we have the historical daily data for the stocks
    4. If we don't have the data, we will get the data

    Initialize the web socket with Alpaca API:
    1. Connect to the web socket
    2. Subscribe to the stocks in the watchlist
    3. Listen for updates on the stocks in the watchlist
    4. Every 30 minute interval we will run the model to get a prediction
    5. Once the buy condition is met, we will run the model once more right before we send the buy order
       to make sure there weren't any drastic changes in the stock price in the 30 minutes since we last ran the model
    
'''


# Imports

import sys

from alpaca_driver.alpaca_driver import AlpacaDriver
from trendformer.trendformer_driver import TrendformerDriver
from data_manager.data_manager import DataManager
# from notification_manager.notification_manager import NotificationManager
from logger import Logger



# Generate a class for the driver
class Driver:

    def __init__(self):
        # Initialize logger
        self.logger = Logger('main', 'logs/main.log').get_logger()

        # Class variables
        self.watchlist  = []
        self.dataframes = {}

        try:
            # Initialize the drivers
            self.alpaca_driver          = AlpacaDriver()
            self.trendformer_driver     = TrendformerDriver()
            self.data_manager           = DataManager()
            # self.notification_manager = NotificationManager()

            self.watchlist  = self.alpaca_driver.get_watchlist()
            self.dataframes = self.data_manager.get_data(self.watchlist)

            self.logger.info('Initialized successfully')
        except Exception as e:
            self.logger.error(f'An error occurred during initialization: {e}')
            sys.exit(1)


    # Sidenote the data manager should be the one that records things like predictions, buy/sell orders, etc.
    # Since it will be the one that is interfacing with CSV files and it makes most sense to have that type of data stored in a CSV file
    # It will make things so much easier when we go to plot the stock data and the predictions and buy/sell orders on a graph.

    def run(self):
        # Run the websocket
        self.alpaca_driver.initiate_websocket(self.data_handler, self.watchlist)

    async def data_handler(self, data):
        
        # Maybe there needs to be some sort of interval where we run the model every 30 minutes or so instead of every minute.
        # But I dont see a big issue the way it is now since running it once every minute isnt that big of a deal. ALthough it could
        # be interesting to see if I can store a prediction every minute and every 15 minutes use the last 15 predictions majority vote to determine the buy/sell. 
        symbol = data.symbol
        df = self.dataframes[symbol]

        new_row = {
            'date': data.timestamp.strftime('%Y-%m-%d'),
            'Open': data.open,
            'High': data.high,
            'Low': data.low,
            'Close': data.close
        }

        refreshed_df = self.data_manager.refresh_dataframe(df, new_row)

        prediction = self.trendformer_driver.predict(refreshed_df)

        # I have the predictions now. I need to use the data manager to store the prediction in a csv file along
        # along with a timestamp of when it was made and the stock symbol. 

        # So a prediction is made every minute for each stock. 
        # Store the prediction is a dictionary with the key being the stock symbol and the value being a list of predictions.
        # Check to see if the length of the predictions is equal to 15. If it is check to see what the consensus is. 
        # Example if 10/15 predictions is buy then buy if you can. This way we arent relying on a single moment in time to make a decision and 
        # we are taking into account the last 15 minutes of movement.

        # After checking the consensus, clear the list of predictions and start over.

        # The buy and sell logic will be called once we hit 15 predictions.



        # This function will be used to handle the data that is received from the websocket
        # We will use this function to run the model and make predictions
        # We will also use this function to check the buy/sell conditions and send the buy/sell orders
        # We will also use this function to record the predictions, buy/sell orders, etc.

        # Run the model to get the prediction
        # prediction = self.trendformer_driver.run_model(data)

        # Check the buy/sell conditions
        # buy_condition = self._check_buy_condition(prediction)
        # sell_condition = self._check_sell_condition(prediction)

        # If the buy condition is met, send the buy order
        # if buy_condition:
        #     self._send_buy_order()
        #     self._record_buy_order()

        # If the sell condition is met, send the sell order
        # if sell_condition:
        #     self._send_sell_order()
        #     self._record_sell_order()

        # Record the prediction
        # self._record_prediction(prediction)

        # Record the data
        # self._record_data(data)

        # Send a notification
        # self._send_notification()

        # Log the data
        # self._log_data(data)

        # Print the data
        breakpoint()
        print(data)


# generate main function
def main():
    # Initialize the driver
    driver = Driver()

    # Run the driver
    driver.run()


if __name__ == '__main__':
    main()
