'''
    This file contains the AlpacaDriver class, which is used to interact with the Alpaca API.
'''

# Imports
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.stream import TradingStream
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
            self.alpaca_client = TradingClient(
                self.API_KEY, self.SECRET, paper=True)
            AlpacaDriver._instance = self

    # Generate the get_watchlist function

    def get_watchlist(self):
        # The watchlist endpoint is broken so Ill harcode it for now
        # watchlists = self.alpaca_client.get_watchlist_by_name('Primary Watchlist')
        # Need to fetch/populate on init
        watchlist = ['AMZN', 'GOOG', 'PYPL', 'U', 'AAPL']
        return watchlist

    # Generate the get_account function
    def get_account(self):
        # Get the account
        account = self.alpaca_client.get_account()
        # Return the account
        return account

    # Generate the get_orders function
    def get_orders(self):
        # Get the orders
        orders = self.alpaca_client.list_orders()
        # Return the orders
        return orders

    # Generate the get_positions function
    def get_all_positions(self):
        # Get the positions
        positions = self.alpaca_client.get_all_positions()
        # Return the positions
        return positions

    # Generate the get_asset function
    def get_asset(self, ticker):
        # Get the asset
        asset = self.alpaca_client.get_asset(ticker)
        # Return the asset
        return asset


    def send_order(self, symbol='AMZN', limit_price=0, quantity=0, order_type='sell', time_in_force='day'):
        limit_order_data = LimitOrderRequest(
            symbol=symbol,
            limit_price=limit_price,
            qty=quantity,
            side=order_type,
            time_in_force=time_in_force
        )
        
        limit_order = self.alpaca_client.submit_order(
            order_data=limit_order_data
        )
        return limit_order


    def initiate_bar_stream(self, bar_data_handler, watchlist):
        wss_client = StockDataStream(self.API_KEY, self.SECRET)
        wss_client.subscribe_daily_bars(bar_data_handler, *watchlist)
        wss_client.run()

    def initiate_trade_updates_stream(self, trade_update_handler):
        trading_stream = TradingStream(self.API_KEY, self.SECRET, paper=True)
        trading_stream.subscribe_trade_updates(trade_update_handler)
        trading_stream.run()


# Generate main function to initialize the class
def main():
    # Initialize the class
    alpaca = AlpacaDriver()
    watchlists = alpaca.get_watchlist()
    account = alpaca.get_account()
    print(watchlists)
    print(account.buying_power)


if __name__ == '__main__':
    main()
