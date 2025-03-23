import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
from data_collection import DataCollectionAgent

class TradingStrategyAgent:
    """
    Agent responsible for generating trading signals, making trade decisions,
    and executing trades based on technical indicators and strategy rules.
    """
    def __init__(self, min_holding_period=5, max_trades_per_week=1):
        # Risk management parameters: stop loss and take profit thresholds.
        self.stop_loss_threshold = 0.05   # 5% drop triggers stop loss
        self.take_profit_threshold = 0.10 # 10% rise triggers take profit
        
        # Trade management parameters.
        self.min_holding_period = min_holding_period
        self.max_trades_per_week = max_trades_per_week
        self.positions = {}
        self.trades = []
    
    def compute_entry_price(self, analyzed_data):
        """
        Compute the dynamic entry price.
        If the 'BB_Lower' column is available, use its latest value.
        Otherwise, use 98% of the current close price.
        
        Parameters:
            analyzed_data (pd.DataFrame): DataFrame with analyzed price data.
        
        Returns:
            float: The computed entry price.
        """
        last_row = analyzed_data.iloc[-1]
        if 'BB_Lower' in analyzed_data.columns:
            return float(last_row['BB_Lower'])
        else:
            return float(last_row['Close']) * 0.98
    
    def generate_trade_signal(self, analyzed_data):
        """
        Generate a trade signal based on technical indicators.
        
        Parameters:
            analyzed_data (pd.DataFrame): DataFrame with analyzed price data.
            
        Returns:
            tuple: (signal, current_price) where signal is "BUY", "SELL", or "HOLD".
        """
        last_row = analyzed_data.iloc[-1]
        
        # Helper function to safely convert to float.
        def to_float(val):
            try:
                return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
            except Exception:
                return None

        # Use fallback values if technical indicators are missing.
        rsi = to_float(last_row['RSI']) if 'RSI' in last_row else 50.0  # Neutral RSI
        macd = to_float(last_row['MACD']) if 'MACD' in last_row else 0.0
        macd_signal = to_float(last_row['MACD_Signal']) if 'MACD_Signal' in last_row else 0.0
        current_price = to_float(last_row['Close']) if 'Close' in last_row else None
        
        if current_price is None:
            # If current price is missing, we cannot generate a signal.
            return ("HOLD", 0)
        
        # Compute dynamic entry price.
        entry_price = self.compute_entry_price(analyzed_data)
        
        # Determine initial signal based on indicator thresholds.
        if rsi < 30 and macd > macd_signal:
            signal = "BUY"
        elif rsi > 70 and macd < macd_signal:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Apply risk management: override signal to SELL if current price deviates significantly.
        if current_price <= entry_price * (1 - self.stop_loss_threshold):
            signal = "SELL"
        elif current_price >= entry_price * (1 + self.take_profit_threshold):
            signal = "SELL"
        
        return (signal, current_price)

    
    def generate_trade_decision(self, signals_df, ticker, current_date):
        """
        Generate a trade decision based on the provided signals and strategy rules.
        This method checks for sufficient data, applies trade frequency and holding period constraints,
        and then decides whether to BUY, SELL, or HOLD.
        
        Parameters:
            signals_df (pd.DataFrame): DataFrame containing signal data.
            ticker (str): Ticker symbol.
            current_date (datetime): Current simulation date.
            
        Returns:
            dict: Trade decision details.
        """
        if current_date not in signals_df.index:
            return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
        
        day_data = signals_df.loc[current_date]
        
        # Use the 'Signal' column if it exists; default to HOLD otherwise.
        signal_value = 0.0
        if 'Signal' in day_data:
            try:
                signal_value = float(day_data['Signal'])
            except (ValueError, TypeError):
                signal_value = 0.0
        
        try:
            close_price = float(day_data['Close'])
        except (ValueError, TypeError, KeyError):
            close_price = 100.0
        
        in_position = ticker in self.positions
        
        # Count the number of trades in the past week.
        recent_trades = sum(1 for trade in self.trades 
                            if trade['date'] > (current_date - timedelta(days=7)) 
                            and trade['action'] in ['BUY', 'SELL'])
        
        if recent_trades >= self.max_trades_per_week:
            return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
        
        # If already in position, enforce a minimum holding period.
        if in_position:
            days_held = (current_date - self.positions[ticker]['entry_date']).days
            if days_held < self.min_holding_period:
                return {'action': 'HOLD', 'ticker': ticker, 'date': current_date}
        
        decision = {'ticker': ticker, 'date': current_date}
        
        if signal_value > 0.5 and not in_position:
            decision['action'] = 'BUY'
            decision['price'] = close_price
            decision['signal_strength'] = signal_value
        elif signal_value < -0.5 and in_position:
            decision['action'] = 'SELL'
            decision['price'] = close_price
            decision['signal_strength'] = signal_value
            entry_price = self.positions[ticker]['entry_price']
            profit_pct = (close_price - entry_price) / entry_price * 100
            decision['profit_pct'] = profit_pct
        else:
            decision['action'] = 'HOLD'
        
        return decision
    
    def execute_trade(self, decision):
        """
        Execute a trade based on the decision details.
        Updates positions and records the trade.
        
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
                'size': 1  # Trade size can be adjusted as needed.
            }
        elif action == 'SELL':
            if ticker in self.positions:
                entry_price = self.positions[ticker]['entry_price']
                exit_price = decision['price']
                profit_pct = (exit_price - entry_price) / entry_price * 100
                decision['profit_pct'] = profit_pct
                del self.positions[ticker]
            else:
                decision['action'] = 'INVALID'
        self.trades.append(decision)
        return decision
    
    def generate_trade_instructions(self, analyzed_portfolio, trade_date):
        """
        Generate trade instructions for each ticker in the analyzed portfolio.
        For each ticker, the method uses data up to the given trade date to compute a signal.
        
        Parameters:
            analyzed_portfolio (dict): Mapping of ticker symbols to their analyzed DataFrames.
            trade_date (datetime): The date for which to generate instructions.
            
        Returns:
            dict: Mapping of ticker to a tuple (signal, current_price).
        """
        instructions = {}
        for ticker, df in analyzed_portfolio.items():
            if trade_date in df.index:
                subset = df.loc[:trade_date]
                signal, price = self.generate_trade_signal(subset)
                instructions[ticker] = (signal, price)
        return instructions

if __name__ == "__main__":
    # Define the ticker and date range for data collection.
    ticker = "SPY"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    
    # Initialize the data collection agent and collect market data.
    data_agent = DataCollectionAgent()
    market_data = data_agent.collect_data(ticker)
    
    # Check if market data was successfully collected.
    if market_data.empty:
        print("Market data collection failed. Please check the ticker or date range.")
    else:
        # For demonstration, create a signals column (dummy signal values) if not already present.
        if 'Signal' not in market_data.columns:
            market_data['Signal'] = np.random.uniform(-1, 1, len(market_data))
        # For demonstration, also add dummy technical indicators if missing.
        for col in ['RSI', 'MACD', 'MACD_Signal']:
            if col not in market_data.columns:
                market_data[col] = np.random.uniform(20, 80, len(market_data)) if col == 'RSI' else np.random.randn(len(market_data))
        # For demonstration, add BB_Lower if not available.
        if 'BB_Lower' not in market_data.columns:
            market_data['BB_Lower'] = market_data['Close'] * 0.95
        
        # Create an analyzed portfolio dictionary.
        analyzed_portfolio = {ticker: market_data}
        
        # Initialize the trading strategy agent.
        strategy_agent = TradingStrategyAgent()
        
        # Generate a trade signal using the collected market data.
        trade_signal, current_price = strategy_agent.generate_trade_signal(market_data)
        print("Trade Signal:", trade_signal, "at price:", current_price)
        
        # Generate trade instructions for the latest available trade date.
        trade_date = market_data.index[-1]
        instructions = strategy_agent.generate_trade_instructions(analyzed_portfolio, trade_date)
        print("Trade Instructions on", trade_date.date(), ":", instructions)
        
        # Generate a trade decision and execute the trade.
        decision = strategy_agent.generate_trade_decision(market_data, ticker, trade_date)
        print("Trade Decision:", decision)
        
        executed_decision = strategy_agent.execute_trade(decision)
        print("Executed Trade:", executed_decision)
