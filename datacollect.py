import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import logging
import warnings
import requests
from dotenv import load_dotenv, find_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

# Set up logging
log_dir = os.path.join("output")
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
            outlier_count = int(outliers.sum())
        
            if outlier_count > 0:
                logger.info(f"Detected {outlier_count} outliers in {ticker} {column} data")
            
                # Calculate rolling median
                rolling_median = df[column].rolling(window=5, center=True).median()
            
                # Replace outliers using np.where to avoid multidimensional indexing issues
                df[column] = np.where(outliers, rolling_median, df[column])
            
                logger.info("Replaced outliers with rolling median")
    
    def get_financial_ratios(self, ticker):
        """
        Retrieve financial ratios for a given ticker from Financial Modeling Prep.
        
        Parameters:
            ticker (str): Stock ticker symbol.
        
        Returns:
            list or None: The retrieved financial ratios data, or None if retrieval fails.
        """
        api_key = os.environ.get('FMP_API_KEY', 'YOUR_FMP_API_KEY')
        url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            ratios_data = response.json()
            print(f"Financial ratios for {ticker} retrieved successfully.")
            return ratios_data
        except Exception as e:
            print("Error fetching financial ratios:", e)
            return None

    def get_risk_free_rate(self):
        """
        Retrieve the latest risk-free rate from the FRED API (using series 'DGS10').
        
        Returns:
            float or None: The latest risk-free rate value, or None if retrieval fails.
        """
        api_key = os.environ.get('FRED_API_KEY', 'YOUR_FRED_API_KEY')
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        try:
            response = requests.get(url)
            response.raise_for_status()
            fred_data = response.json()
            observations = fred_data.get("observations", [])
            if observations:
                latest_observation = observations[0]
                value = latest_observation.get("value", "0")
                if value == ".":
                    print("Latest risk-free rate value is missing in FRED data.")
                    return None
                risk_free_rate = float(value)
                print("Latest risk-free rate retrieved successfully.")
                return risk_free_rate
            else:
                print("No observations found in FRED response.")
                return None
        except Exception as e:
            print("Error fetching risk free rate from FRED:", e)
            return None

    def get_available_etfs(self):
        """
        Retrieve an up-to-date list of available ETFs from Financial Modeling Prep.
        
        Returns:
            list: A list of available ETF ticker symbols.
        """
        api_key = os.environ.get('FMP_API_KEY', 'YOUR_FMP_API_KEY')
        url = f"https://financialmodelingprep.com/api/v3/etf/list?apikey={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            etfs = response.json()
            available_tickers = [item.get("symbol") for item in etfs if "symbol" in item and item.get("symbol")]
            print(f"Retrieved {len(available_tickers)} available ETFs successfully.")
            return available_tickers
        except Exception as e:
            print("Error fetching available ETFs:", e)
            return []
