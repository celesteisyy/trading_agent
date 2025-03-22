#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 15:34:56 2025

@author: wodewenjianjia
"""
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingSystem")

class DataCollectionAgent:
    """
    Agent responsible for gathering market data, ensuring quality,
    and preprocessing for analysis.
    """
    
    def __init__(self):
        self.data = {}
        logger.info("Data Collection Agent initialized")
    
    def fetch_data(self, tickers, start_date, end_date=None):
        """
        Fetch historical market data for specified tickers
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols to fetch
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format, defaults to today
        
        Returns:
        --------
        dict
            Dictionary of DataFrames with ticker as key
        """
        # Ensure dates are strings for yfinance
        if not isinstance(start_date, str):
            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        elif not isinstance(end_date, str):
            if hasattr(end_date, 'strftime'):
                end_date = end_date.strftime('%Y-%m-%d')
            
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
            
        for ticker in tickers:
            try:
                # Use a more defensive approach when downloading data
                data = yf.download(ticker, start=start_date, end=end_date)
                
                # Verify we have data and it's not empty
                if data is not None and not data.empty:
                    self.data[ticker] = data
                    logger.info(f"Successfully fetched data for {ticker}: {len(data)} records")
                else:
                    logger.warning(f"No data returned for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
        
        return self.data
    
    def handle_missing_data(self, method='ffill'):
        """
        Handle missing values in the dataset
        
        Parameters:
        -----------
        method : str
            Method to fill missing values ('ffill', 'bfill', 'interpolate')
        """
        for ticker in self.data:
            # Check for missing values
            missing_count = self.data[ticker].isna().sum().sum()
            if missing_count > 0:
                logger.info(f"Handling {missing_count} missing values for {ticker} using {method}")
                
                if method == 'ffill':
                    self.data[ticker].fillna(method='ffill', inplace=True)
                elif method == 'bfill':
                    self.data[ticker].fillna(method='bfill', inplace=True)
                elif method == 'interpolate':
                    self.data[ticker].interpolate(method='linear', inplace=True)
    
    def remove_outliers(self, column='Close', threshold=3):
        """
        Detect and handle outliers in the data
    
        Parameters:
        -----------
        column : str
            Column to check for outliers
        threshold : float
            Z-score threshold for outlier detection
        """
        for ticker in self.data:
            df = self.data[ticker]
        
        # Check if column exists
            if column not in df.columns:
               logger.warning(f"Column {column} not found in data for {ticker}")
               continue
            
        # Calculate z-scores
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        
        # Create boolean mask for outliers using comparison
            outliers = z_scores > threshold
        
        # Count outliers using .sum() method - Convert to scalar with int()
            outlier_count = int(outliers.sum())  # Convert to scalar integer
        
            if outlier_count > 0:  # Now comparing scalar to scalar
               logger.info(f"Detected {outlier_count} outliers in {ticker} {column} data")
            
            # Calculate rolling median
               rolling_median = df[column].rolling(window=5, center=True).median()
            
            # Replace outliers using boolean indexing
               df.loc[outliers, column] = rolling_median.loc[outliers]
            
               logger.info("Replaced outliers with rolling median")