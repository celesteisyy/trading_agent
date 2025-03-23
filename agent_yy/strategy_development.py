import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging to both file and console.
log_dir = os.path.join("agent_yy", "output")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "trading_system.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingSystem")

class StrategyAgent:
    """
    Agent responsible for trade strategy execution, signal generation, and rules.
    This version integrates the simple signal generation logic that was originally in the analysis module.
    """
    def __init__(self, min_holding_period=5, max_trades_per_week=1):
        self.min_holding_period = min_holding_period
        self.max_trades_per_week = max_trades_per_week
        self.positions = {}  # Track current positions.
        self.trades = []     # Record executed trades.
        logger.info("Strategy Agent initialized")
    
    def generate_signals(self, data, ticker):
        """
        Integrated signal generation function.
        This function computes a 20-day simple moving average (SMA) for the 'Close' price.
        It then generates a 'Signal' column:
          - If the Close is above the SMA, signal is 1 (BUY).
          - If the Close is below the SMA, signal is -1 (SELL).
          - Otherwise, 0 (HOLD).
          
        Parameters:
            data (pd.DataFrame): Historical price data for the given ticker.
            ticker (str): Ticker symbol.
        
        Returns:
            pd.DataFrame: DataFrame with an added 'Signal' column.
        """
        if 'Close' not in data.columns:
            logger.warning(f"Data for {ticker} does not contain 'Close' prices.")
            data['Signal'] = 0
            return data

        data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        # Initialize Signal column with 0
        data['Signal'] = 0
        data.loc[data['Close'] > data['SMA_20'], 'Signal'] = 1
        data.loc[data['Close'] < data['SMA_20'], 'Signal'] = -1
        logger.info(f"Generated signals for {ticker} using 20-day SMA")
        return data

    def generate_trade_decisions(self, signals_df, ticker, current_date):
        """
        Generate trade decisions based on signals and strategy rules.
        
        Parameters:
            signals_df (pd.DataFrame): DataFrame with signals and technical indicator data.
            ticker (str): Ticker symbol.
            current_date (datetime): The current simulation date.
            
        Returns:
            dict: Trade decision with action and metadata.
        """
        # Before generating decisions, ensure signals are up-to-date.
        # If the 'Signal' column is missing, generate signals using the integrated function.
        if 'Signal' not in signals_df.columns:
            signals_df = self.generate_signals(signals_df, ticker)
        
        if current_date not in signals_df.index:
            logger.warning(f"No data for {current_date} in signals dataframe for {ticker}")
            return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
        
        day_data = signals_df.loc[current_date]
        
        # Retrieve the signal value; if conversion fails, default to 0.
        try:
            signal_value = float(day_data['Signal'])
        except (ValueError, TypeError):
            logger.warning(f"Could not convert signal to float for {ticker} on {current_date}")
            signal_value = 0.0
        
        # Retrieve close price.
        try:
            close_price = float(day_data['Close'])
        except (ValueError, TypeError, KeyError):
            logger.warning(f"Could not get closing price for {ticker} on {current_date}")
            close_price = 100.0  # Fallback placeholder
        
        in_position = ticker in self.positions
        
        # Check trade frequency constraints.
        recent_trades = sum(
            1 for trade in self.trades 
            if trade['date'] > (current_date - timedelta(days=7)) 
            and trade['action'] in ['BUY', 'SELL']
        )
        if recent_trades >= self.max_trades_per_week:
            logger.info(f"Maximum trades per week reached ({self.max_trades_per_week}) for {ticker}, holding position")
            return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
        
        # Enforce minimum holding period.
        if in_position:
            days_held = (current_date - self.positions[ticker]['entry_date']).days
            if days_held < self.min_holding_period:
                logger.info(f"Minimum holding period not reached for {ticker} ({days_held}/{self.min_holding_period} days), holding position")
                return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
        
        decision = {'ticker': ticker, 'date': current_date}
        if signal_value > 0 and not in_position:
            decision['action'] = 'BUY'
            decision['price'] = close_price
            decision['signal_strength'] = signal_value
            logger.info(f"BUY signal for {ticker} at {close_price}")
        elif signal_value < 0 and in_position:
            decision['action'] = 'SELL'
            decision['price'] = close_price
            decision['signal_strength'] = signal_value
            entry_price = self.positions[ticker]['entry_price']
            profit_pct = (close_price - entry_price) / entry_price * 100
            decision['profit_pct'] = profit_pct
            logger.info(f"SELL signal for {ticker} at {close_price} (P&L: {profit_pct:.2f}%)")
        else:
            decision['action'] = 'HOLD'
            logger.info(f"HOLD signal for {ticker}, current signal value: {signal_value:.2f}")
        
        return decision
    
    def execute_trade(self, decision):
        """
        Execute a trade based on the decision.
        
        Parameters:
            decision (dict): Trade decision details.
            
        Returns:
            dict: Updated decision with execution details.
        """
        ticker = decision['ticker']
        action = decision['action']
        
        if action == 'BUY':
            self.positions[ticker] = {
                'entry_price': decision['price'],
                'entry_date': decision['date'],
                'size': 1  # This can be updated by the portfolio manager.
            }
            logger.info(f"Executed BUY for {ticker} at {decision['price']}")
        elif action == 'SELL':
            if ticker in self.positions:
                entry_price = self.positions[ticker]['entry_price']
                exit_price = decision['price']
                profit_pct = (exit_price - entry_price) / entry_price * 100
                decision['profit_pct'] = profit_pct
                del self.positions[ticker]
                logger.info(f"Executed SELL for {ticker} at {exit_price} (P&L: {profit_pct:.2f}%)")
            else:
                logger.warning(f"Attempted to SELL {ticker} but no position exists")
                decision['action'] = 'INVALID'
        
        self.trades.append(decision)
        return decision

# For testing purposes, you can add a main section if desired.
if __name__ == "__main__":
    # Create dummy data for testing.
    test_data = pd.DataFrame({
        'Close': [100, 102, 105, 103, 107, 110],
    }, index=pd.date_range(start='2023-01-01', periods=6, freq='D'))
    
    # Remove any existing Signal column to test integrated generation.
    if 'Signal' in test_data.columns:
        del test_data['Signal']
    
    agent = StrategyAgent()
    # Generate signals using integrated method.
    test_data = agent.generate_signals(test_data, "TEST")
    print("Data with generated signals:")
    print(test_data.tail())
    
    # Assume we want a decision on the last date.
    decision = agent.generate_trade_decisions(test_data, "TEST", test_data.index[-1])
    print("Trade Decision:", decision)
    
    execution = agent.execute_trade(decision)
    print("Execution Result:", execution)
