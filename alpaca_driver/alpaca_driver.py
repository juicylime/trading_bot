'''
    This file contains the AlpacaDriver class, which is used to interact with the Alpaca API.
'''

# Imports
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus
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
            self.trading_client = TradingClient(
                self.API_KEY, self.SECRET, paper=True)
            self.stock_data_stream = StockDataStream(self.API_KEY, self.SECRET)
            self.trading_stream = TradingStream(
                self.API_KEY, self.SECRET, paper=True)
            AlpacaDriver._instance = self

    # Generate the get_watchlist function

    def get_watchlist(self):
        # The watchlist endpoint is broken so Ill harcode it for now
        # watchlists = self.trading_client.get_watchlist_by_name('Primary Watchlist')
        # Need to fetch/populate on init
        watchlist = ['AMZN', 'GOOG', 'PYPL', 'U', 'AAPL']
        return watchlist

    # Generate the get_account function
    def get_account(self):
        # Get the account
        account = self.trading_client.get_account()
        # Return the account
        return account

    # Generate the get_orders function
    def get_orders(self, type='buy'):
        # Get the orders
        # params to filter orders by
        side = OrderSide.BUY if type == 'buy' else OrderSide.SELL

        request_params = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            side=side
        )

        orders = self.trading_client.get_orders(filter=request_params)
        # Return the orders
        return orders

    # Generate the get_positions function
    def get_all_positions(self):
        # Get the positions
        positions = self.trading_client.get_all_positions()
        # Return the positions
        return positions

    # Generate the get_asset function
    def get_asset(self, ticker):
        # Get the asset
        asset = self.trading_client.get_asset(ticker)
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

        limit_order = self.trading_client.submit_order(
            order_data=limit_order_data
        )
        return limit_order

    def cancel_all_orders(self):
        cancel_statuses = self.trading_client.cancel_orders()
        return cancel_statuses

    def start_bar_stream(self, bar_data_handler, watchlist):
        self.stock_data_stream.subscribe_daily_bars(
            bar_data_handler, *watchlist)
        self.stock_data_stream.run()

    def start_trade_updates_stream(self, trade_update_handler):
        self.trading_stream.subscribe_trade_updates(trade_update_handler)
        self.trading_stream.run()

    def close_bar_stream(self):
        self.stock_data_stream.stop()

    def close_trade_updates_stream(self):
        self.trading_stream.stop()


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
