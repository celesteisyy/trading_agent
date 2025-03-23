import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AnalysisAgent:
    """
    Agent responsible for technical analysis and generating signals.
    """
    def __init__(self):
        self.indicators = {}
    
    def calculate_indicators(self, data, ticker):
        """
        Calculate technical indicators for a given ticker.
        Adds moving averages, RSI, MACD, Bollinger Bands, and ATR.
        """
        df = data.copy()
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        
        # Average True Range (ATR)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        self.indicators[ticker] = df
        return df
    
    def generate_signals_very_simple(self, ticker):
        """
        Generate simple random trading signals for testing purposes.
        """
        if ticker not in self.indicators:
            print(f"No indicator data available for {ticker}")
            return None
            
        df = self.indicators[ticker].copy()
        import random
        df['Signal'] = [random.uniform(-1, 1) for _ in range(len(df))]
        self.indicators[ticker] = df
        return df
    
    def analyze_portfolio(self, portfolio_data):
        """
        Analyze each asset's data in a portfolio.
        Returns a dictionary mapping tickers to analyzed DataFrames.
        """
        analyzed = {}
        for ticker, df in portfolio_data.items():
            analyzed[ticker] = self.calculate_indicators(df, ticker)
        return analyzed

if __name__ == "__main__":
    # Test analysis on dummy data.
    dates = pd.date_range(start="2022-01-01", periods=50, freq='B')
    dummy_data = pd.DataFrame({
        'Open': np.random.randn(50).cumsum() + 100,
        'High': np.random.randn(50).cumsum() + 102,
        'Low': np.random.randn(50).cumsum() + 98,
        'Close': np.random.randn(50).cumsum() + 100,
        'Volume': np.random.randint(1000000, 2000000, size=50)
    }, index=dates)
    agent = AnalysisAgent()
    analyzed = agent.calculate_indicators(dummy_data, "SPY")
    print("Analyzed Data:")
    print(analyzed.head())
