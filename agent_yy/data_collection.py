import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

class DataCollectionAgent:
    """
    Agent responsible for gathering market data, ensuring quality,
    and preprocessing data for analysis.
    """
    def __init__(self):
        self.data = {}
    
    def collect_data(self, ticker, start_date, end_date):
        """
        Collect historical market data for a given ticker and date range using yfinance,
        then perform quality checks.
        """
        print(f"Collecting market data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        data = self.quality_check(data)
        return data

    def quality_check(self, data):
        """
        Perform quality checks:
         - Handle MultiIndex by extracting the date level.
         - Ensure the index is a DatetimeIndex.
         - Forward-fill missing values.
         - Remove outliers where price deviates more than 10% from the mean.
        """
        if isinstance(data.index, pd.MultiIndex):
            data.index = data.index.get_level_values(1)
        data.index = pd.to_datetime(data.index)
        data = data.ffill()
        for col in ['Open', 'High', 'Low', 'Close']:
            mean_val = data[col].mean()
            mask = abs(data[col] - mean_val) / mean_val > 0.1
            data[col] = data[col].mask(mask).ffill()
        return data

    def get_financial_ratios(self, ticker):
        """
        Retrieve financial ratios for a given ticker from Financial Modeling Prep.
        """
        url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={os.environ.get('FMP_API_KEY','YOUR_FMP_API_KEY')}"
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
        """
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={os.environ.get('FRED_API_KEY','YOUR_FRED_API_KEY')}&file_type=json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            fred_data = response.json()
            observations = fred_data.get("observations", [])
            if observations:
                latest_observation = observations[-1]
                risk_free_rate = float(latest_observation.get("value", 0))
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
        """
        url = f"https://financialmodelingprep.com/api/v3/etf/list?apikey={os.environ.get('FMP_API_KEY','YOUR_FMP_API_KEY')}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            etfs = response.json()
            available_tickers = [item.get("symbol") for item in etfs if "symbol" in item]
            print("Available ETFs retrieved successfully.")
            return available_tickers
        except Exception as e:
            print("Error fetching available ETFs:", e)
            return []

if __name__ == "__main__":
    # Test DataCollectionAgent functionality.
    agent = DataCollectionAgent()
    ticker = "SPY"
    data = agent.collect_data(ticker, "2022-01-01", "2022-12-31")
    print("Sample Market Data:")
    print(data.head())
    available_etfs = agent.get_available_etfs()
    print("\nAvailable ETFs:", available_etfs)
    ratios = agent.get_financial_ratios("AAPL")
    if ratios:
        print("\nFinancial ratios for AAPL:")
        print(ratios)
    risk_free_rate = agent.get_risk_free_rate()
    if risk_free_rate is not None:
        print("\nLatest risk-free rate (10-Year Treasury):", risk_free_rate)
