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
import time as t
from datetime import datetime, time
import pytz
import traceback
import threading
import json
import os

from alpaca_driver.alpaca_driver import AlpacaDriver
from trendformer.trendformer_driver import TrendformerDriver
from data_manager.data_manager import DataManager
from notification_manager.notification_manager import NotificationManager
from logger import Logger


# Generate a class for the driver
class Driver:
    """
    The Driver class is responsible for managing the trading bot's operations.
    It handles the initialization, starting and ending of the trading day, as well as
    handling bar data updates and making buy/sell decisions based on predictions and
    other criteria.
    """

    def __init__(self):
        """
        Initializes the trading bot driver.

        Attributes:
        - logger: The logger object for logging messages.
        - watchlist: The list of stocks to watch.
        - dataframes: A dictionary to store dataframes for each stock.
        - predictions: A dictionary to store predictions for each stock.
        - scores: A dictionary to store prediction scores for each stock.
        - todays_data: A dictionary to store today's data for each stock.
        - positions: A list to store current positions.
        - goal_hit: A dictionary to track if the goal has been hit for each stock.
        - buy_orders: A list to keep track of open buy orders.
        - buying_power: The available buying power.
        - time_since_entry: A dictionary to keep track of how long we've been in a position for each stock.
        - prediction_interval: The interval in minutes for analyzing predictions.
        - prediction_threshold: The threshold for considering a prediction as positive.
        - goal_threshold: The threshold for the goal as a percentage of the Average True Range (ATR).
        - hard_stop_loss_threshold: The threshold for the hard stop loss as a multiple of the ATR.
        - trailing_stop_loss_threshold: The threshold for the trailing stop loss as a percentage of the ATR.
        - consensus_weight: The weight for consensus in making predictions.
        - distance_below_ema_weight: The weight for distance below Exponential Moving Average (EMA) in making predictions.
        - consensus_made: The number of consensus made for the watchlist.
        - alpaca_driver: The Alpaca driver object for interacting with the Alpaca API.
        - trendformer_driver: The Trendformer driver object for making stock predictions.
        - data_manager: The data manager object for managing stock data.
        - notification_manager: The notification manager object for sending notifications.
        - trade_updates_thread: The thread for receiving trade updates.
        - bar_update_thread: The thread for receiving bar updates.
        """
        self.prediction_logger = Logger('predictions', 'logs/prediction.log').get_logger()
        self.debug_logger = Logger('debug', 'logs/debug.log').get_logger()
        self.trade_logger = Logger('trades', 'logs/trade.log').get_logger()
        self.watchlist = []
        self.dataframes = {}
        self.predictions = {}
        self.scores = {}
        self.todays_data = {}
        self.positions = []
        self.goal_hit = {}
        self.buy_orders = []
        self.buying_power = 0
        self.prediction_interval = 15  # minutes
        self.prediction_threshold = 0.5
        self.goal_threshold = 0.8
        self.hard_stop_loss_threshold = 2
        self.trailing_stop_loss_threshold = 0.2
        self.consensus_weight = 1
        self.distance_below_ema_weight = 1
        self.consensus_made = 0
        self.alpaca_driver = AlpacaDriver()
        self.trendformer_driver = TrendformerDriver()
        self.data_manager = DataManager()
        self.notification_manager = NotificationManager()
        self.trade_updates_thread = None
        self.bar_update_thread = None

        if os.path.exists('time_since_entry.json'):
            with open('time_since_entry.json', 'r') as f:
                self.time_since_entry = json.load(f)
        else:
            self.time_since_entry = {}
            with open('time_since_entry.json', 'w') as f:
                json.dump(self.time_since_entry, f)

    def start_day(self):
        """
        Starts the trading day by initializing necessary data, updating daily data,
        and starting the trading threads.
        """
        try:
            self.debug_logger.info('Starting day')
            self.buying_power = float(
                self.alpaca_driver.get_account().buying_power)
            self.positions = self.alpaca_driver.get_all_positions()
            self.watchlist = self.alpaca_driver.get_watchlist()
            self.data_manager.update_daily_data(self.watchlist)
            self.increment_time_since_entry()

            self.dataframes = self.data_manager.get_data(self.watchlist)
            self.predictions = {symbol: {'up': [], 'down': [],
                                         'consensus': 0} for symbol in self.watchlist}
            self.scores = {symbol: 0 for symbol in self.watchlist}

            self.debug_logger.info('Started day successfully')
        except Exception as e:
            msg = 'An error occurred during start of day:'
            self.debug_logger.error(
                f'{msg} {e}\nStack Trace: {traceback.format_exc()}')
            self.notification_manager.send_error_alert(
                {'title': msg, 'traceback': traceback.format_exc()})
            sys.exit(1)

        self.start_trading()

    def end_day(self):
        """
        Ends the trading day by stopping the trading threads and canceling all open orders.
        """
        try:
            self.debug_logger.info('Ending day')
            self.save_state()
            self.stop_trading()
            self.cancel_all_orders()
            self.debug_logger.info('Ended day successfully')
        except Exception as e:
            msg = 'An error occurred during end of day:'
            self.debug_logger.error(
                f'{msg} {e}\nStack Trace: {traceback.format_exc()}')
            self.notification_manager.send_error_alert(
                {'title': msg, 'traceback': traceback.format_exc()})
            
            self.save_state()
            sys.exit(1)

    def start_trading(self):
        """
        Starts the trading threads for receiving bar data updates and trade updates.
        """
        self.trade_updates_thread = threading.Thread(
            target=self.alpaca_driver.start_trade_updates_stream, args=(self.trade_update_handler,))
        self.trade_updates_thread.start()

        self.bar_update_thread = threading.Thread(
            target=self.alpaca_driver.start_bar_stream, args=(self.bar_data_handler, self.watchlist))
        self.bar_update_thread.start()

    def increment_time_since_entry(self):
        """
        Increments the time since entry for each position in the trading bot.
        """
        for symbol in self.time_since_entry:
            self.time_since_entry[symbol] += 1

    def stop_trading(self):
        """
        Stops the trading threads for receiving bar data updates and trade updates.
        """
        self.alpaca_driver.close_bar_stream()
        self.alpaca_driver.close_trade_updates_stream()

    def cancel_all_orders(self):
        """
        Cancels all open orders.
        """
        cancel_statuses = self.alpaca_driver.cancel_all_orders()
        self.debug_logger.info(
            f'Attempt to cancel all open orders: {cancel_statuses}')

    def cancel_buy_orders(self):
        """
        Cancels all the buy orders placed by the trading bot.
        """
        for order in self.buy_orders:
            self.alpaca_driver.cancel_order(order.id)
            self.debug_logger.info(
                f'Attempt to cancel buy order {order.id}')
            
    def save_state(self):
        """
        Saves the state of the trading bot.
        Just 1 variable but if you need to save anything else, youd do it here.
        """
        with open('time_since_entry.json', 'w') as f:
            json.dump(self.time_since_entry, f)

    async def bar_data_handler(self, data):
        """
        Handles the bar data updates and makes buy/sell decisions based on predictions and other criteria.
        """
        """
            TODO: The buy and sell logic should exist in its own class.
                  It should implement the strategy pattern.
                  There should be a Trader class that has a few basic functions like buy, sell and cancel orders.
                  There will be trading strategy classes that implement the buy and sell logic that the trader will use to 
                  buy and sell stocks.
                  The strategies will implement and interface that has a buy and sell function. Or maybe a check_buy and check_sell function.
                  The trader will call the check_buy and check_sell functions and if they return true, the trader will call the buy and sell functions.
        
                  This class  could actually be the trader class. It's already designed for it
        """
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
                    # Every 15 minutes cancel the existing buy orders.
                    self.cancel_buy_orders()

                    self.prediction_logger.info(
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
                        self.trade_logger.info(
                            f'Hard stop loss triggered. Sending sell order for {symbol} at {current_price}')

                    # Trailing stop loss (We manually setup the trailing stop loss feature to provide extra flexibility down the road)
                    # If the current price is 0.2 x the ATR below the goal, sell
                    if self.goal_hit[symbol] == True:
                        if current_price <= entry_price - (self.trailing_stop_loss_threshold * avg_true_range):
                            self.alpaca_driver.send_order(
                                symbol=symbol, limit_price=current_price, quantity=current_position.qty, order_type='sell', time_in_force='day')
                            self.trade_logger.info(
                                f'Trailing stop loss triggered. Sending sell order for {symbol} at {current_price}')

                    # If we've been in the position for 3 days and the consensus is below the threshold, sell
                    if self.time_since_entry[symbol] >= 3:
                        if self.predictions[symbol]['consensus'] < self.prediction_threshold:
                            self.alpaca_driver.send_order(
                                symbol=symbol, limit_price=current_price, quantity=current_position.qty, order_type='sell', time_in_force='day')
                            self.trade_logger.info(
                                f'3 days passed, new prediction threshold triggered. Sending sell order for {symbol} at {current_price}')

                        else:
                            self.time_since_entry[symbol] = 0

                elif self.consensus_made == len(self.watchlist):
                    self.positions = self.alpaca_driver.get_all_positions()
                    self.buy_orders = self.alpaca_driver.get_orders(type='buy')

                    if len(self.positions) + len(self.buy_orders) < 3:
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
                            if len(self.positions) + len(self.buy_orders) == 0:
                                buying_power = self.buying_power * 0.3
                            elif len(self.positions) + len(self.buy_orders) == 1:
                                buying_power = self.buying_power * 0.5
                            elif len(self.positions) + len(self.buy_orders) == 2:
                                buying_power = self.buying_power * 1.0
                            elif len(self.positions) + len(self.buy_orders) >= 3:
                                buying_power = 0

                            quantity = buying_power // symbol_price

                            if quantity > 0:
                                if self.predictions[symbol]['consensus'] > self.prediction_threshold:
                                    if symbol_price < self.todays_data[symbol]['EMA_10']:
                                        self.alpaca_driver.send_order(
                                            symbol=symbol, limit_price=symbol_price, quantity=quantity, order_type='buy', time_in_force='day')
                                        self.notification_manager.send_trade_alert({'update_title': 'Send Buy Order', 'details': {
                                            'symbol': symbol, 'side': 'buy', 'quantity': quantity}})
                                        self.trade_logger.info(
                                            f'Sending buy order for {symbol} | Price: {symbol_price} | Prediction: {self.predictions[symbol]["consensus"]}')

                    # Reset the consensus counter. We want to have a new set of consensus for the watchlist.
                    self.consensus_made = 0
                
        except Exception as e:
            msg = 'An error occurred during trading:'
            self.debug_logger.error(
                f'{msg} {e}\nStack Trace: {traceback.format_exc()}')
            self.notification_manager.send_error_alert(
                {'title': msg, 'traceback': traceback.format_exc()})
            
            # Even though we arent exiting the program, we still want to save the state just in case..
            self.save_state()

    async def trade_update_handler(self, data):
        """
        Handles trade updates received from the trading platform.

        Args:
            data (object): The trade update data.

        Raises:
            Exception: If an error occurs during the trade update handler.

        Returns:
            None
        """
        try:
            # Send notification here. Need to see how the order object looks like first.
            if data.event == 'fill' or data.event == 'partial_fill' or data.event == 'canceled' or data.event == 'rejected':
                self.notification_manager.send_trade_alert({'update_title': 'Order Update', 'details': {
                    'symbol': data.order.symbol, 'side': data.order.side, 'quantity': data.order.qty}})

            if data.event == 'fill':
                self.data_manager.store_trade(
                    data.order.symbol, data.order.side, data.order.qty, float(data.order.filled_avg_price), self.predictions[data.order.symbol]['consensus'], self.alpaca_driver.get_account().equity)

            self.trade_logger.info(
                f'Order update: {data.order.symbol} | {data.order.side} | {data.order.qty}')
            # change self.buy_orders and self.positions accordingly
            self.positions = self.alpaca_driver.get_all_positions()
            self.buy_orders = self.alpaca_driver.get_orders(type='buy')

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
            self.debug_logger.error(
                f'{msg} {e}\nStack Trace: {traceback.format_exc()}')
            self.notification_manager.send_error_alert(
                {'title': msg, 'traceback': traceback.format_exc()})
            
            self.save_state()
            sys.exit(1)


def main():
    # Initialize the driver
    print('Starting driver')
    driver = Driver()

    # Define the time when the tasks should run in New York time
    start_time_ny = time(9, 30)  # 9:30 AM
    end_time_ny = time(16, 0)  # 4:00 PM
    print(f'Start NY time: {start_time_ny}, End NY time: {end_time_ny}')

    # Get the current date and time in New York
    new_york_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(new_york_tz)
    print(f'Current NY time: {now_ny}')

    # Convert the New York times to the local timezone
    local_tz = pytz.timezone('America/Vancouver')
    start_time_local = new_york_tz.localize(datetime.combine(
        now_ny.date(), start_time_ny)).astimezone(local_tz).time()
    end_time_local = new_york_tz.localize(datetime.combine(
        now_ny.date(), end_time_ny)).astimezone(local_tz).time()
    print(
        f'Start local time: {start_time_local}, End local time: {end_time_local}')

    # Schedule the tasks to run at the converted times
    schedule.every().monday.at(start_time_local.strftime("%H:%M")).do(driver.start_day)
    schedule.every().tuesday.at(start_time_local.strftime("%H:%M")).do(driver.start_day)
    schedule.every().wednesday.at(start_time_local.strftime("%H:%M")).do(driver.start_day)
    schedule.every().thursday.at(start_time_local.strftime("%H:%M")).do(driver.start_day)
    schedule.every().friday.at(start_time_local.strftime("%H:%M")).do(driver.start_day)

    schedule.every().monday.at(end_time_local.strftime("%H:%M")).do(driver.end_day)
    schedule.every().tuesday.at(end_time_local.strftime("%H:%M")).do(driver.end_day)
    schedule.every().wednesday.at(end_time_local.strftime("%H:%M")).do(driver.end_day)
    schedule.every().thursday.at(end_time_local.strftime("%H:%M")).do(driver.end_day)
    schedule.every().friday.at(end_time_local.strftime("%H:%M")).do(driver.end_day)

    print('Starting scheduler')

    while True:
        schedule.run_pending()
        t.sleep(1)

if __name__ == '__main__':
    main()
