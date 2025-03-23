import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta

class StrategyDevelopmentAgent:
    """
    Agent responsible for generating trading signals and trade instructions.
    """
    def __init__(self):
        # Risk management parameters.
        self.stop_loss_threshold = 0.05   # 5%
        self.take_profit_threshold = 0.10 # 10%
    
    def compute_entry_price(self, analyzed_data):
        """
        Dynamically compute the entry price.
        Uses Bollinger Band lower band if available; otherwise, returns 98% of the current close price.
        """
        last_row = analyzed_data.iloc[-1]
        if 'BB_Lower' in analyzed_data.columns:
            bb_lower = last_row['BB_Lower']
            return float(bb_lower.iloc[0]) if hasattr(bb_lower, 'iloc') else float(bb_lower)
        else:
            return float(last_row['Close']) * 0.98
    
    def generate_trade_signal(self, analyzed_data):
        """
        Generate a trade signal based on technical indicators.
        Steps:
          1. Extract the last row of the analyzed DataFrame.
          2. Convert RSI, MACD, Signal_Line, and current price to float.
          3. Compute the dynamic entry price.
          4. Determine the signal based on indicator thresholds.
          5. Apply risk management: if current price deviates significantly from entry price, signal SELL.
        Returns:
          (signal, current_price)
        """
        last_row = analyzed_data.iloc[-1]
        def to_float(val):
            return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
        rsi = to_float(last_row['RSI'])
        macd = to_float(last_row['MACD'])
        signal_line = to_float(last_row['MACD_Signal'])
        current_price = to_float(last_row['Close'])
        entry_price = self.compute_entry_price(analyzed_data)
        if rsi < 30 and macd > signal_line:
            signal = "BUY"
        elif rsi > 70 and macd < signal_line:
            signal = "SELL"
        else:
            signal = "HOLD"
        if signal == "HOLD":
            if current_price <= entry_price * (1 - self.stop_loss_threshold):
                signal = "SELL"
            elif current_price >= entry_price * (1 + self.take_profit_threshold):
                signal = "SELL"
        return signal, current_price
    
    def generate_trade_instructions(self, analyzed_portfolio, trade_date):
        """
        Given an analyzed portfolio (a dictionary mapping tickers to analyzed DataFrames)
        and a trade_date, return a dictionary mapping each ticker to a tuple (signal, price).
        """
        instructions = {}
        for ticker, df in analyzed_portfolio.items():
            if trade_date in df.index:
                subset = df.loc[:trade_date]
                signal, price = self.generate_trade_signal(subset)
                instructions[ticker] = (signal, price)
        return instructions

if __name__ == "__main__":
    # Test StrategyDevelopmentAgent on dummy analyzed data.
    dates = pd.date_range(start="2022-01-01", periods=50, freq='B')
    dummy_data = pd.DataFrame({
        'RSI': np.random.uniform(20, 80, 50),
        'MACD': np.random.randn(50),
        'MACD_Signal': np.random.randn(50),
        'Close': np.random.randn(50).cumsum() + 100,
        'BB_Lower': np.random.randn(50).cumsum() + 90
    }, index=dates)
    agent = StrategyDevelopmentAgent()
    signal, price = agent.generate_trade_signal(dummy_data)
    print("Trade Signal:", signal, "at price:", price)
    analyzed_portfolio = {"SPY": dummy_data}
    trade_date = dummy_data.index[-1]
    instructions = agent.generate_trade_instructions(analyzed_portfolio, trade_date)
    print("Trade Instructions on", trade_date.date(), ":", instructions)
