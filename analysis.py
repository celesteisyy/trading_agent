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

# Set up logging
log_dir = os.path.join("agent_lc", "output")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingSystem")


class AnalysisAgent:
    """
    Agent responsible for technical analysis and generating signals
    """
    
    def __init__(self):
        self.indicators = {}
        logger.info("Analysis Agent initialized")
    
    def calculate_indicators(self, data, ticker):
        """
        Calculate technical indicators for a given ticker
        
        Parameters:
        -----------
        data : DataFrame
            Price data for a ticker
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        DataFrame
            Data with added technical indicators
        """
        logger.info(f"Calculating technical indicators for {ticker}")
        df = data.copy()
        
        # 1. Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # 2. Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 4. Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        
        # 5. Average True Range (ATR)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Store indicators for this ticker
        self.indicators[ticker] = df
        logger.info(f"Successfully calculated indicators for {ticker}")
        
        return df
    
    def generate_signals_very_simple(self, ticker):
        """
        Generate very simple trading signals using random values
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        DataFrame
            Data with added signal columns
        """
        if ticker not in self.indicators:
            logger.error(f"No indicator data available for {ticker}")
            return None
            
        # Get the DataFrame and make a fresh copy
        df = self.indicators[ticker].copy()
        logger.info(f"Generating very simple trading signals for {ticker}")
        
        # Create a list of signal values with some random buy/sell signals
        import random
        signals = []
        
        # Process each row separately with some random values
        for i in range(len(df)):
            # Generate a random value between -1 and 1
            signal = random.uniform(-1, 1)
            signals.append(signal)
        
        # Add the Signal column all at once
        df['Signal'] = signals
        
        # Explicitly store the updated DataFrame back in the indicators dictionary
        self.indicators[ticker] = df
        
        # Print some debug info
        logger.info(f"Successfully added Signal column to {ticker} data")
        logger.info(f"First few signal values for {ticker}: {df['Signal'].head().tolist()}")
        
        return df
    
    def generate_signals(self, ticker):
        """
        Generate trading signals based on indicators
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        DataFrame
            Data with added signal columns
        """
        if ticker not in self.indicators:
            logger.error(f"No indicator data available for {ticker}")
            return None
            
        df = self.indicators[ticker].copy()
        logger.info(f"Generating trading signals for {ticker}")
        
        # Initialize signal columns
        df['MA_Signal'] = 0
        df['RSI_Signal'] = 0
        df['MACD_Cross_Signal'] = 0
        df['BB_Signal'] = 0
        df['Signal'] = 0
        
        # Process signals row by row to avoid alignment issues
        for i in range(len(df)):
            # Get all values as scalars to avoid Series comparison
            try:
                # Safe extraction of values with error handling
                row = df.iloc[i]
                
                # 1. Moving Average Crossover
                if not pd.isna(row['SMA_20']) and not pd.isna(row['SMA_50']):
                    sma20 = float(row['SMA_20'])
                    sma50 = float(row['SMA_50'])
                    if sma20 > sma50:
                        df.iloc[i, df.columns.get_loc('MA_Signal')] = 1
                    elif sma20 < sma50:
                        df.iloc[i, df.columns.get_loc('MA_Signal')] = -1
                
                # 2. RSI Oversold/Overbought
                if not pd.isna(row['RSI']):
                    rsi = float(row['RSI'])
                    if rsi < 30:
                        df.iloc[i, df.columns.get_loc('RSI_Signal')] = 1
                    elif rsi > 70:
                        df.iloc[i, df.columns.get_loc('RSI_Signal')] = -1
                
                # 3. MACD Signal
                if not pd.isna(row['MACD']) and not pd.isna(row['MACD_Signal']):
                    macd = float(row['MACD'])
                    macd_signal = float(row['MACD_Signal'])
                    if macd > macd_signal:
                        df.iloc[i, df.columns.get_loc('MACD_Cross_Signal')] = 1
                    elif macd < macd_signal:
                        df.iloc[i, df.columns.get_loc('MACD_Cross_Signal')] = -1
                
                # 4. Bollinger Band Bounce
                if not pd.isna(row['Close']) and not pd.isna(row['BB_Lower']) and not pd.isna(row['BB_Upper']):
                    close = float(row['Close'])
                    bb_lower = float(row['BB_Lower'])
                    bb_upper = float(row['BB_Upper'])
                    if close < bb_lower:
                        df.iloc[i, df.columns.get_loc('BB_Signal')] = 1
                    elif close > bb_upper:
                        df.iloc[i, df.columns.get_loc('BB_Signal')] = -1
                
                # Combine signals (weighted average)
                ma_signal = float(df.iloc[i, df.columns.get_loc('MA_Signal')])
                macd_signal = float(df.iloc[i, df.columns.get_loc('MACD_Cross_Signal')])
                rsi_signal = float(df.iloc[i, df.columns.get_loc('RSI_Signal')])
                bb_signal = float(df.iloc[i, df.columns.get_loc('BB_Signal')])
                
                combined_signal = (
                    0.3 * ma_signal +
                    0.3 * macd_signal + 
                    0.2 * rsi_signal +
                    0.2 * bb_signal
                )
                
                df.iloc[i, df.columns.get_loc('Signal')] = combined_signal
                
            except Exception as e:
                # Skip problematic rows but log the error
                logger.warning(f"Error processing row {i} for {ticker}: {str(e)}")
                continue
        
        logger.info(f"Signals generated for {ticker}")
        return df