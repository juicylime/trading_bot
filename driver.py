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
import asyncio
import heapq

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
        self.watchlist = []
        self.dataframes = {}
        self.predictions = {}
        self.scores = {}
        self.todays_data = {}
        self.positions = []

        self.prediction_interval = 1  # minutes
        self.prediction_threshold = 0.5

        self.consensus_weight = 1
        self.distance_below_ema_weight = 1

        # Keep track of how many consesus were made for the watchlist
        self.consensus_made = 0

        try:
            # Initialize the drivers
            self.alpaca_driver = AlpacaDriver()
            self.trendformer_driver = TrendformerDriver()
            self.data_manager = DataManager()
            # self.notification_manager = NotificationManager()

            self.watchlist = self.alpaca_driver.get_watchlist()
            self.dataframes = self.data_manager.get_data(self.watchlist)
            self.predictions = {symbol: {'up': [], 'down': [],
                                         'consensus': 0} for symbol in self.watchlist}
            self.scores = {symbol: 0 for symbol in self.watchlist}


            self.logger.info('Initialized successfully')
        except Exception as e:
            self.logger.error(f'An error occurred during initialization: {e}')
            sys.exit(1)

    # Sidenote the data manager should be the one that records things like predictions, buy/sell orders, etc.
    # Since it will be the one that is interfacing with CSV files and it makes most sense to have that type of data stored in a CSV file
    # It will make things so much easier when we go to plot the stock data and the predictions and buy/sell orders on a graph.

    def run(self):
        # Initiate the websockets
        self.alpaca_driver.initiate_bar_stream(
            self.bar_data_handler, self.watchlist)
        self.alpaca_driver.initiate_trade_updates_stream(
            self.trade_update_handler)

    async def bar_data_handler(self, data):
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
        today = refreshed_df.iloc[-1]
        
        self.todays_data[symbol] = today


        positions = self.alpaca_driver.get_all_positions()
        if symbol in positions:
            # Generate code to sell the stock
            # self.alpaca_driver.send_order(...)
            # self.notification_manager.send_notification(...)
            pass

        
        prediction = self.trendformer_driver.predict(refreshed_df)

        # Ill be storing the predictions when I buy or sell too so when I plot it ill be able to tell what the prediction was at the time.
        lock = asyncio.Lock()

        async with lock:
            if (len(self.predictions[symbol]['up']) + len(self.predictions[symbol]['down'])) == self.prediction_interval:
                self.data_manager.store_prediction(symbol, prediction)

                # The Consensus will be the average of the majority
                if len(self.predictions[symbol]['up']) > len(self.predictions[symbol]['down']):
                    consensus = sum(
                        self.predictions[symbol]['up']) / len(self.predictions[symbol]['up'])
                else:
                    consensus = sum(
                        self.predictions[symbol]['down']) / len(self.predictions[symbol]['down'])

                # Store the consensus and clear the predictions
                self.predictions[symbol]['consensus'] = consensus
                self.predictions[symbol]['up'] = []
                self.predictions[symbol]['down'] = []

                if self.consensus_made == len(self.watchlist):
                    self.consensus_made = 1

                self.consensus_made += 1

                below_ema_score = (
                    (today['Close'] - today['EMA_10']) / today['EMA_10']) * 100
                self.scores[symbol] = (self.consensus_weight * consensus) + \
                    (self.distance_below_ema_weight * below_ema_score)

            else:
                if prediction > self.prediction_threshold:
                    self.predictions[symbol]['up'].append(prediction)
                else:
                    self.predictions[symbol]['down'].append(prediction)

            buying_power = float(self.alpaca_driver.get_account().buying_power)
            

            # Also I need to restrict my trading from market open to market close. My buys should be only be open for a few minutes.

            # I need to add the function in data manager to add a new row every day.



            '''
                Things to do:
                1. Restrict trading to market open to market close. Dont place any orders outside of that time frame.
                2. Add a new row to the historical data every day.
                3. Create a seperate function to figure out how much buying power to use.
                4. Create the sell logic.
                       - Hard Stop loss at around 2-3 x the ATR
                       - Trailing stop loss at around 0.2 - 0.5 x the ATR after the stock has gone up 0.8 - 1.5 x the ATR
                       - Every 3 days after entering the position, use the current prediction to sell if its a downtrend or Hold 
                       for another 3 days if its an uptrend.
            '''

            #Logic to divide the buying power based on how many positions we have
            # I need a better way to figure out how much buying power to use
            if len(positions) == 0:
                buying_power = buying_power * 0.3
            elif len(positions) == 1:
                buying_power = buying_power * 0.5
            elif len(positions)[symbol] == 2:
                buying_power = buying_power * 1.0
            elif len(positions)[symbol] == 3:
                buying_power = 0

            quantity = buying_power // today['Close']


            # I want to clean this up to make these conditions more readable. Long if checks are horrible

            # If all consensus have been made and we have enough buying power we can look into buying
            # if self.consensus_made == len(self.watchlist) and quantity > 0:
            if quantity > 0:
                # Get the stocks that we can afford
                affordable_stocks = {symbol: self.scores[symbol]
                                for symbol, today in self.todays_data.items() if today['Close'] < buying_power}
                
                top_3_symbols = heapq.nlargest(3, affordable_stocks, key=affordable_stocks.get)

                if symbol in top_3_symbols:
                    if self.predictions[symbol]['consensus'] > self.prediction_threshold and today['Close'] < today['EMA_10']*2:
                        self.positions.append(symbol)
                        self.alpaca_driver.send_order(
                            symbol=symbol, limit_price=today['Close'], quantity=quantity, order_type='buy', time_in_force='day')
                        self.logger.info(f'Buying {symbol} at {today["Close"]}')
                    # This is where we send a notification. Also another one when the order is filled.
                    # self.notification_manager.send_notification(...)


        # Log the data
        # self._log_data(data)

        # Print the data
        print(data)

    async def trade_update_handler(self, data):
        # Just for logging and sending notifications
        # log whatever the outcome of the order is and send a notification
        # Also store in data manager too
        print(data)

# generate main function


def main():
    # Initialize the driver
    driver = Driver()

    # Run the driver
    driver.run()


if __name__ == '__main__':
    main()
