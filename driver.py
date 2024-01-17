''' 
    This file will serve as the main driver for the application
    The application is a trading bot that will use the Alpaca API to trade stocks
    The bot will use a trained model to predict the value of the 5 day moving average 3 days into the future

    What the bot will do
    
'''


# Imports
import sys
import asyncio
import heapq
import schedule
import time
from datetime import datetime
import pytz
import traceback
import threading

from alpaca_driver.alpaca_driver import AlpacaDriver
from trendformer.trendformer_driver import TrendformerDriver
from data_manager.data_manager import DataManager
from notification_manager.notification_manager import NotificationManager
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
        self.goal_hit = {}
        self.buy_orders = 0  # Keep track of how many buy orders are open.
        self.buying_power = 0

        self.time_since_entry = {}  # Keep track of how long we've been in a position

        self.prediction_interval = 15  # minutes
        self.prediction_threshold = 0.5

        self.goal_threshold = 0.8  # 80% of the ATR
        # 2 x the ATR <-- I can be more aggressive here since this is a hard stop loss. Maybe 4 x the ATR?
        self.hard_stop_loss_threshold = 2
        self.trailing_stop_loss_threshold = 0.2  # 20% of the ATR

        self.consensus_weight = 1
        self.distance_below_ema_weight = 1

        # Keep track of how many consesus were made for the watchlist
        self.consensus_made = 0

        self.alpaca_driver = AlpacaDriver()
        self.trendformer_driver = TrendformerDriver()
        self.data_manager = DataManager()
        self.notification_manager = NotificationManager()

        self.trade_updates_thread = None

    def start_day(self):
        try:
            self.logger.info('Starting day')
            self.buying_power = float(
                self.alpaca_driver.get_account().buying_power)
            self.positions = self.alpaca_driver.get_all_positions()
            self.watchlist = self.alpaca_driver.get_watchlist()
            self.data_manager.update_daily_data(self.watchlist)

            self.dataframes = self.data_manager.get_data(self.watchlist)
            self.predictions = {symbol: {'up': [], 'down': [],
                                         'consensus': 0} for symbol in self.watchlist}
            self.scores = {symbol: 0 for symbol in self.watchlist}

            self.logger.info('Started day successfully')
        except Exception as e:
            msg = 'An error occurred during start of day:'
            self.logger.error(f'{msg} {e}')
            self.notification_manager.send_error_alert(
                {'title': msg, 'traceback': traceback.format_exc()})
            sys.exit(1)

        self.start_trading()

    def end_day(self):
        try:
            self.logger.info('Ending day')
            self.stop_trading()
            self.cancel_all_orders()
            self.logger.info('Ended day successfully')
        except Exception as e:
            msg = 'An error occurred during end of day:'
            self.logger.error(f'{msg} {e}')
            self.notification_manager.send_error_alert(traceback.format_exc())
            sys.exit(1)

    # Keep the bar updates on the main thread and the trade updates on a separate thread
    # The close function should stop the thread too hopefully
    def start_trading(self):
        self.trade_updates_thread = threading.Thread(
            target=self.alpaca_driver.start_trade_updates_stream, args=(self.trade_update_handler,))
        self.trade_updates_thread.start()

        self.alpaca_driver.start_bar_stream(
            self.bar_data_handler, self.watchlist)

    def increment_time_since_entry(self):
        for symbol in self.time_since_entry:
            self.time_since_entry[symbol] += 1

    def stop_trading(self):
        self.alpaca_driver.close_bar_stream()
        self.alpaca_driver.close_trade_updates_stream()

    def cancel_all_orders(self):
        cancel_statuses = self.alpaca_driver.cancel_all_orders()
        self.logger.info(
            f'Attempt to cancel all open orders: {cancel_statuses}')

    async def bar_data_handler(self, data):
        try:
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
            current_price = round(today['Close'], 2)

            prediction = self.trendformer_driver.predict(refreshed_df)

            lock = asyncio.Lock()

            async with lock:
                if (len(self.predictions[symbol]['up']) + len(self.predictions[symbol]['down'])) == self.prediction_interval:
                    self.logger.info(
                        f'Storing prediction for {symbol} | Prediction: {prediction}')
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
                        self.consensus_made = 0

                    self.consensus_made += 1

                    below_ema_score = (
                        (current_price - today['EMA_10']) / today['EMA_10']) * 100
                    self.scores[symbol] = (self.consensus_weight * consensus) + \
                        (self.distance_below_ema_weight * below_ema_score)

                else:
                    if prediction > self.prediction_threshold:
                        self.predictions[symbol]['up'].append(prediction)
                    else:
                        self.predictions[symbol]['down'].append(prediction)

                current_position = next(
                    (position for position in self.positions if position.symbol == symbol), None)

                # If we have a current position we only look at sell logic, otherwise we look at buy logic
                # To keep it simple initially.
                if current_position is not None:
                    # If its a new position
                    if symbol not in self.time_since_entry:
                        self.time_since_entry[symbol] = 0

                    if symbol not in self.goal_hit:
                        self.goal_hit[symbol] = False

                    entry_price = float(current_position.avg_entry_price)
                    avg_true_range = today['ATRr_14']

                    if current_price >= entry_price + (self.goal_threshold * avg_true_range):
                        self.goal_hit[symbol] = True

                    # Hard stop loss
                    # If the current price is 2 x the ATR below the entry price, sell
                    if current_price <= entry_price - (self.hard_stop_loss_threshold * avg_true_range):
                        self.alpaca_driver.send_order(
                            symbol=symbol, limit_price=current_price, quantity=current_position.qty, order_type='sell', time_in_force='day')
                        self.logger.info(
                            f'Hard stop loss triggered. Sending sell order for {symbol} at {current_price}')
                        self.notification_manager.send_trade_alert({'update_title': 'Send Sell Order', 'details': {
                            'symbol': symbol, 'side': 'sell', 'quantity': current_position.qty}})

                    # Trailing stop loss (We manually setup the trailing stop loss feature to provide extra flexibility down the road)
                    # If the current price is 0.2 x the ATR below the goal, sell
                    if self.goal_hit[symbol] == True:
                        if current_price <= entry_price - (self.trailing_stop_loss_threshold * avg_true_range):
                            self.alpaca_driver.send_order(
                                symbol=symbol, limit_price=current_price, quantity=current_position.qty, order_type='sell', time_in_force='day')
                            self.logger.info(
                                f'Trailing stop loss triggered. Sending sell order for {symbol} at {current_price}')
                            self.notification_manager.send_trade_alert({'update_title': 'Send Sell Order', 'details': {
                                'symbol': symbol, 'side': 'sell', 'quantity': current_position.qty}})

                    # If we've been in the position for 3 days and the consensus is below the threshold, sell
                    if self.time_since_entry[symbol] >= 3:
                        if self.predictions[symbol]['consensus'] < self.prediction_threshold:
                            self.alpaca_driver.send_order(
                                symbol=symbol, limit_price=current_price, quantity=current_position.qty, order_type='sell', time_in_force='day')
                            self.logger.info(
                                f'3 days passed, new prediction threshold triggered. Sending sell order for {symbol} at {current_price}')
                            self.notification_manager.send_trade_alert({'update_title': 'Send Sell Order', 'details': {
                                'symbol': symbol, 'side': 'sell', 'quantity': current_position.qty}})

                        else:
                            self.time_since_entry[symbol] = 0

                elif self.consensus_made == len(self.watchlist):
                    self.positions = self.alpaca_driver.get_all_positions()
                    self.buy_orders = len(
                        self.alpaca_driver.get_orders(type='buy'))

                    if len(self.positions) + self.buy_orders < 3:
                        # Get the stocks that we can afford
                        positions_symbols = {
                            position.symbol for position in self.positions}
                        sorted_scores = {symbol: score for symbol, score in sorted(self.scores.items(
                        ), key=lambda item: item[1], reverse=True) if symbol not in positions_symbols}

                        for symbol in sorted_scores:
                            symbol_price = round(
                                self.todays_data[symbol]['Close'])

                            self.buying_power = float(
                                self.alpaca_driver.get_account().buying_power)

                            # Logic to divide the buying power based on how many positions we have
                            if len(self.positions) + self.buy_orders == 0:
                                buying_power = self.buying_power * 0.3
                            elif len(self.positions) + self.buy_orders == 1:
                                buying_power = self.buying_power * 0.5
                            elif len(self.positions) + self.buy_orders == 2:
                                buying_power = self.buying_power * 1.0
                            elif len(self.positions) + self.buy_orders >= 3:
                                buying_power = 0

                            quantity = buying_power // symbol_price

                            if quantity > 0:
                                if self.predictions[symbol]['consensus'] > self.prediction_threshold:
                                    if symbol_price < self.todays_data[symbol]['EMA_10'] * 2:
                                        self.alpaca_driver.send_order(
                                            symbol=symbol, limit_price=symbol_price, quantity=quantity, order_type='buy', time_in_force='day')
                                        self.notification_manager.send_trade_alert({'update_title': 'Send Buy Order', 'details': {
                                            'symbol': symbol, 'side': 'buy', 'quantity': quantity}})
                                        self.logger.info(
                                            f'Sending buy order for {symbol} | Price: {symbol_price} | Prediction: {self.predictions[symbol]["consensus"]}')

                    # Reset the consensus counter. We want to have a new set of consensus for the watchlist.
                    self.consensus_made = 0

            # Print the data
            print(data)
        except Exception as e:
            msg = 'An error occurred during trading:'
            self.logger.error(f'{msg} {e}')
            self.notification_manager.send_error_alert(
                {'title': msg, 'traceback': traceback.format_exc()})
            sys.exit(1)

    async def trade_update_handler(self, data):
        try:
            # Send notification here. Need to see how the order object looks like first.
            if data.event == 'fill' or data.event == 'partial_fill' or data.event == 'canceled' or data.event == 'rejected':
                self.notification_manager.send_trade_alert({'update_title': 'Order Update', 'details': {
                    'symbol': data.symbol, 'side': data.side, 'quantity': data.qty}})

            if data.event == 'fill':
                self.data_manager.store_trade(
                    data.symbol, data.side, data.qty, data.price, self.predictions[data.symbol]['consensus'])
                
            self.logger.info(
                f'Order update: {data.symbol} | {data.side} | {data.qty}')
            # change self.buy_orders and self.positions accordingly
            self.positions = self.alpaca_driver.get_all_positions()
            self.buy_orders = len(self.alpaca_driver.get_orders(type='buy'))

            # Get a list of symbols from the positions
            position_symbols = [position.symbol for position in self.positions]
            symbols_to_remove = [
                symbol for symbol in self.time_since_entry if symbol not in position_symbols]

            # Remove the symbols from time_since_entry
            for symbol in symbols_to_remove:
                del self.time_since_entry[symbol]

            # Update the buying power
            self.buying_power = float(
                self.alpaca_driver.get_account().buying_power)
        except Exception as e:
            msg = 'An error occurred during trade update handler:'
            self.logger.error(f'{msg} {e}')
            self.notification_manager.send_error_alert(
                {'title': msg, 'traceback': traceback.format_exc()})
            sys.exit(1)


def main():
    # Initialize the driver
    print('Starting driver')
    driver = Driver()
    # driver.start_day()
    # Get the current time in your desired timezone
    new_york_tz = pytz.timezone('America/New_York')
    new_york_time = datetime.now(new_york_tz)

    # Get the current time in your system's local timezone
    # Get the current time in your system's local timezone
    local_tz = pytz.timezone('America/Vancouver')
    local_time = datetime.now(local_tz)

    # Calculate the difference in hours between the two timezones
    time_difference = (new_york_time - local_time).seconds // 3600

    # Schedule the job based on the time difference
    schedule.every().day.at(
        f"{(9 + time_difference) % 24:02d}:30").do(driver.start_day)
    schedule.every().day.at(
        f"{(16 + time_difference) % 24:02d}:00").do(driver.end_day)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()
