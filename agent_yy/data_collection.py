import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

class DataCollectionAgent:
    """
    Agent responsible for gathering market data, ensuring quality,
    and preprocessing data for analysis.
    The time window is set internally to the past 5 years.
    """
    def __init__(self):
        self.data = {}
    
    def collect_data(self, ticker):
        """
        Collect historical market data for a given ticker using yfinance over the past 5 years,
        then perform quality checks.
        
        Parameters:
            ticker (str): Stock ticker symbol.
        
        Returns:
            pd.DataFrame: The quality-checked market data.
        """
        end_date = datetime.today()
        start_date = end_date - relativedelta(years=5)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        print(f"Collecting market data for {ticker} from {start_date_str} to {end_date_str}...")
        data = yf.download(ticker, start=start_date_str, end=end_date_str)
        
        if not data.empty:
            data = self.quality_check(data)
            self.data[ticker] = data
        else:
            print(f"No data found for {ticker} in the specified date range.")
            
        return data

    def quality_check(self, data):
        """
        Perform quality checks:
        - Handle MultiIndex by extracting the date level for the index and flattening columns if necessary.
        - Ensure the index is a DatetimeIndex.
        - Forward-fill missing values.
        - Remove outliers where price deviates more than 10% from the mean.
        """
        data = data.copy()
        
        # Flatten the index if it's a MultiIndex.
        if isinstance(data.index, pd.MultiIndex):
            data.index = data.index.get_level_values(1)
            
        # Flatten columns if they are a MultiIndex.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.index = pd.to_datetime(data.index)
        data = data.ffill()
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                # Convert the column to numeric (this should now be a 1-d Series).
                data[col] = pd.to_numeric(data[col], errors='coerce')
                mean_val = data[col].mean()
                # Check that mean_val is valid and nonzero.
                if pd.notna(mean_val) and mean_val != 0:
                    mask = abs(data[col] - mean_val) / mean_val > 0.1
                    data.loc[mask, col] = np.nan
                    data[col] = data[col].ffill().bfill()
        
        return data

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

if __name__ == "__main__":
    # Test DataCollectionAgent functionality.
    agent = DataCollectionAgent()
    ticker = "SPY"
    data = agent.collect_data(ticker)
    print("Sample Market Data:")
    print(data.head())
    available_etfs = agent.get_available_etfs()
    print("\nAvailable ETFs sample:", available_etfs[:5] if available_etfs else "None")
    ratios = agent.get_financial_ratios("AAPL")
    if ratios:
        print("\nFinancial ratios sample for AAPL:")
        print(ratios[:2] if isinstance(ratios, list) else "Not a list")
    risk_free_rate = agent.get_risk_free_rate()
    if risk_free_rate is not None:
        print("\nLatest risk-free rate (10-Year Treasury):", risk_free_rate)
