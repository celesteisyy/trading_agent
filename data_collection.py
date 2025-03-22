import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

class DataCollectionAgent:
    def __init__(self, fmp_api_key=None, fred_api_key=None):
        """
        Initialize the Data Collection Agent.
        API keys are loaded from environment variables if not provided.
        """
        self.fmp_api_key = fmp_api_key or os.environ.get("FMP_API_KEY", "YOUR_FMP_API_KEY")
        self.fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY", "YOUR_FRED_API_KEY")

    def collect_data(self, ticker, start_date, end_date):
        """
        Collect market data for the given ticker and date range using yfinance.
        Then perform quality checks on the data.
        """
        print(f"Collecting market data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        data = self.quality_check(data)
        return data

    def quality_check(self, data):
        """
        Perform quality checks on the data:
          - If the index is a MultiIndex, extract the date level.
          - Ensure the index is a DatetimeIndex.
          - Forward-fill missing values.
          - Remove outliers: if a price deviates more than 10% from the mean, treat it as an outlier.
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
        url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={self.fmp_api_key}"
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
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={self.fred_api_key}&file_type=json"
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
        url = f"https://financialmodelingprep.com/api/v3/etf/list?apikey={self.fmp_api_key}"
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

    def get_dividend_info(self, symbol):
        """
        Retrieve the most recent dividend payment amount for a given symbol.
        """
        dividend_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}?apikey={self.fmp_api_key}"
        try:
            dividend_response = requests.get(dividend_url)
            dividend_response.raise_for_status()
            dividend_data = dividend_response.json()
            if "historical" in dividend_data and len(dividend_data["historical"]) > 0:
                most_recent_dividend = dividend_data["historical"][0]["adjDividend"]
                print(f"The most recent dividend payment for {symbol} is: ${most_recent_dividend:,.2f}")
                return most_recent_dividend
            else:
                print(f"No dividend data available for {symbol}.")
                return None
        except Exception as e:
            print("Error fetching dividend info:", e)
            return None

    def get_net_income(self, symbol):
        """
        Retrieve the net income from the income statement for a given symbol.
        """
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?apikey={self.fmp_api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data and len(data) > 0:
                net_income = data[0].get("netIncome", None)
                print(f"Net income for {symbol} retrieved successfully.")
                return net_income
            else:
                print(f"No income statement data available for {symbol}.")
                return None
        except Exception as e:
            print("Error fetching net income:", e)
            return None

if __name__ == "__main__":
    # Test the Data Collection Agent functionality.
    agent = DataCollectionAgent()
    
    # Test market data collection using yfinance.
    ticker = "SPY"
    data = agent.collect_data(ticker, "2022-01-01", "2022-12-31")
    print("Sample Market Data:")
    print(data.head())
    
    # Test available ETFs retrieval.
    available_etfs = agent.get_available_etfs()
    print("\nAvailable ETFs:", available_etfs)
    
    # Test financial ratios retrieval.
    ratios = agent.get_financial_ratios("AAPL")
    if ratios:
        print("\nFinancial ratios for AAPL:")
        print(ratios)
    
    # Test risk-free rate retrieval.
    risk_free_rate = agent.get_risk_free_rate()
    if risk_free_rate is not None:
        print("\nLatest risk-free rate (10-Year Treasury):", risk_free_rate)
    
    # Test dividend info retrieval.
    dividend = agent.get_dividend_info("AAPL")
    if dividend is not None:
        print(f"\nMost recent dividend for AAPL: ${dividend:,.2f}")
    
    # Test net income retrieval.
    net_income = agent.get_net_income("AAPL")
    if net_income is not None:
        print(f"\nNet income for AAPL: ${net_income:,.2f}")
