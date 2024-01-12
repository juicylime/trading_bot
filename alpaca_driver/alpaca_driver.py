'''
    This file contains the AlpacaDriver class, which is used to interact with the Alpaca API.
'''

# Imports
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from dotenv import load_dotenv
import os

load_dotenv()

class AlpacaDriver:
    _instance = None

    def __init__(self):
        if AlpacaDriver._instance:
            self = AlpacaDriver._instance
        else:
            self.API_KEY = os.getenv('ALPACA_KEY')
            self.SECRET = os.getenv('ALPACA_SECRET')
            self.alpaca_client = TradingClient(self.API_KEY, self.SECRET, paper=True)
            AlpacaDriver._instance = self

    # Generate the get_data function
    def get_data(self, start_date, end_date, time_frame, ticker):
        # Get the data
        data = self.alpaca.get_barset(
            ticker,
            time_frame,
            start=start_date,
            end=end_date
        ).df
        # Return the data
        return data

    # Generate the get_watchlist function
    def get_watchlist(self):
        # The watchlist endpoint is broken so Ill harcode it for now
        # watchlists = self.alpaca_client.get_watchlist_by_name('Primary Watchlist')
        watchlist = ['AMZN', 'GOOG', 'PYPL', 'U', 'AAPL'] # Need to fetch/populate on init
        return watchlist

    # Generate the get_account function
    def get_account(self):
        # Get the account
        account = self.alpaca_client.get_account()
        # Return the account
        return account

    # Generate the get_position function
    def get_position(self, ticker):
        # Get the position
        position = self.alpaca.get_position(ticker)
        # Return the position
        return position

    # Generate the get_orders function
    def get_orders(self):
        # Get the orders
        orders = self.alpaca.list_orders()
        # Return the orders
        return orders

    # Generate the get_positions function
    def get_positions(self):
        # Get the positions
        positions = self.alpaca.list_positions()
        # Return the positions
        return positions

    # Generate the get_asset function
    def get_asset(self, ticker):
        # Get the asset
        asset = self.alpaca.get_asset(ticker)
        # Return the asset
        return asset
    

    # data_handler is a function that will be used to handle
    # the data that is received from the websocket. We've define that
    # in the main driver file (driver.py)
    def initiate_websocket(self, data_handler, watchlist):
        wss_client = StockDataStream(self.API_KEY, self.SECRET)
        wss_client.subscribe_daily_bars(data_handler, *watchlist)
        wss_client.run()
        


# Generate main function to initialize the class 
# def main():
#     # Initialize the class
#     alpaca = AlpacaDriver()
#     watchlists = alpaca.get_watchlist()
#     account = alpaca.get_account()
#     print(watchlists)
#     print(account)

# if __name__ == '__main__':
#     main()