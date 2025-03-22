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

log_dir = os.path.join("agent_lc", "output")
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
    Agent responsible for trade strategy execution and rules
    """
    
    def __init__(self, min_holding_period=5, max_trades_per_week=1):
        self.min_holding_period = min_holding_period
        self.max_trades_per_week = max_trades_per_week
        self.positions = {}
        self.trades = []
        logger.info("Strategy Agent initialized")
    
    def generate_trade_decisions(self, signals_df, ticker, current_date):
        """
        Generate trade decisions based on signals and strategy rules
        
        Parameters:
        -----------
        signals_df : DataFrame
            DataFrame with signals
        ticker : str
            Ticker symbol
        current_date : datetime
            Current simulation date
            
        Returns:
        --------
        dict
            Trade decision with action and metadata
        """
        # Get signals for current date
        if current_date not in signals_df.index:
            logger.warning(f"No data for {current_date} in signals dataframe")
            return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
        
        day_data = signals_df.loc[current_date]
        
        # Check if 'Signal' column exists, if not, default to HOLD
        if 'Signal' not in day_data:
            logger.warning(f"No 'Signal' column found for {ticker} on {current_date}")
            return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
            
        # Try to convert the signal to a float, with error handling
        try:
            signal_value = float(day_data['Signal'])
        except (ValueError, TypeError):
            # If signal can't be converted to float, use a default value
            logger.warning(f"Could not convert signal to float for {ticker} on {current_date}")
            signal_value = 0.0
        
        # Try to get close price with error handling
        try:
            close_price = float(day_data['Close'])
        except (ValueError, TypeError, KeyError):
            logger.warning(f"Could not get closing price for {ticker} on {current_date}")
            # Default to previous known price or a placeholder
            close_price = 100.0  # Placeholder price
        
        # Check if we're already in a position for this ticker
        in_position = ticker in self.positions
        
        # Check trade frequency constraints
        recent_trades = sum(1 for trade in self.trades 
                            if trade['date'] > (current_date - timedelta(days=7)) 
                            and trade['action'] in ['BUY', 'SELL'])
        
        if recent_trades >= self.max_trades_per_week:
            logger.info(f"Maximum trades per week reached ({self.max_trades_per_week}), holding position")
            return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
        
        # Check minimum holding period
        if in_position:
            days_held = (current_date - self.positions[ticker]['entry_date']).days
            if days_held < self.min_holding_period:
                logger.info(f"Minimum holding period not reached for {ticker} ({days_held}/{self.min_holding_period} days), holding position")
                return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
        
        # Generate decision based on signal
        decision = {'ticker': ticker, 'date': current_date}
        
        if signal_value > 0.5 and not in_position:
            decision['action'] = 'BUY'
            decision['price'] = close_price
            decision['signal_strength'] = signal_value
            logger.info(f"BUY signal for {ticker} at {close_price}")
            
        elif signal_value < -0.5 and in_position:
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
        Execute a trade based on the decision
        
        Parameters:
        -----------
        decision : dict
            Trade decision details
            
        Returns:
        --------
        dict
            Updated decision with execution details
        """
        ticker = decision['ticker']
        action = decision['action']
        
        if action == 'BUY':
            self.positions[ticker] = {
                'entry_price': decision['price'],
                'entry_date': decision['date'],
                'size': 1  # This will be updated by the portfolio manager
            }
            logger.info(f"Executed BUY for {ticker} at {decision['price']}")
            
        elif action == 'SELL':
            if ticker in self.positions:
                # Calculate P&L
                entry_price = self.positions[ticker]['entry_price']
                exit_price = decision['price']
                profit_pct = (exit_price - entry_price) / entry_price * 100
                decision['profit_pct'] = profit_pct
                
                # Record the trade
                del self.positions[ticker]
                logger.info(f"Executed SELL for {ticker} at {exit_price} (P&L: {profit_pct:.2f}%)")
            else:
                logger.warning(f"Attempted to SELL {ticker} but no position exists")
                decision['action'] = 'INVALID'
        
        self.trades.append(decision)
        return decision
