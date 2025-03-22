import pandas as pd
import numpy as np
import yfinance as yf

class DataCollectionAgent:
    def __init__(self):
        # Initialize any data source configuration if needed.
        pass

    def collect_data(self, ticker, start_date, end_date):
        """
        Collect market data for the given ticker and date range using yfinance.
        Then perform quality checks on the data.
        """
        print(f"Collecting data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        data = self.quality_check(data)
        return data

    def quality_check(self, data):
        """
        Perform quality checks on the data:
          - Fill missing values using forward fill.
          - Remove outliers: if a price deviates more than 10% from the mean, treat it as an outlier.
        """
        data = data.fillna(method='ffill')
        for col in ['Open', 'High', 'Low', 'Close']:
            mean_val = data[col].mean()
            data[col] = data[col].apply(lambda x: np.nan if abs(x - mean_val) / mean_val > 0.1 else x)
            data[col] = data[col].fillna(method='ffill')
        return data

if __name__ == "__main__":
    agent = DataCollectionAgent()
    # Example: Collect one year of data for the SPY ETF.
    df = agent.collect_data("SPY", "2022-01-01", "2022-12-31")
    print(df.head())
