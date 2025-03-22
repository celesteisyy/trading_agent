import pandas as pd
from data_collection import DataCollectionAgent

class AnalysisAgent:
    def __init__(self):
        pass

    def calculate_moving_average(self, data, window=20):
        """
        Calculate the moving average of the Close price over a specified window.
        """
        data['MA'] = data['Close'].rolling(window=window).mean()
        return data

    def calculate_RSI(self, data, period=14):
        """
        Calculate the Relative Strength Index (RSI) for the Close price.
        """
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    def calculate_MACD(self, data, short_window=12, long_window=26, signal_window=9):
        """
        Calculate the Moving Average Convergence Divergence (MACD) indicator.
        """
        data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
        data['MACD'] = data['EMA_short'] - data['EMA_long']
        data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
        return data

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """
        Calculate Bollinger Bands for the Close price:
          - BB_MA: the moving average,
          - BB_upper: moving average + (num_std * standard deviation),
          - BB_lower: moving average - (num_std * standard deviation).
        """
        data['BB_MA'] = data['Close'].rolling(window=window).mean()
        data['BB_STD'] = data['Close'].rolling(window=window).std()
        data['BB_upper'] = data['BB_MA'] + num_std * data['BB_STD']
        data['BB_lower'] = data['BB_MA'] - num_std * data['BB_STD']
        return data

    def analyze(self, data):
        """
        Analyze the data by calculating:
          - Moving Average (MA)
          - Relative Strength Index (RSI)
          - MACD and Signal Line
          - Bollinger Bands
        """
        data = self.calculate_moving_average(data)
        data = self.calculate_RSI(data)
        data = self.calculate_MACD(data)
        data = self.calculate_bollinger_bands(data)
        return data

    def analyze_portfolio(self, portfolio_data):
        """
        Analyze a portfolio of ETFs.
        portfolio_data: a dictionary where each key is a ticker and the value is its DataFrame.
        Returns a dictionary with the analyzed DataFrames for each ticker.
        """
        analyzed_portfolio = {}
        for ticker, data in portfolio_data.items():
            analyzed_portfolio[ticker] = self.analyze(data)
        return analyzed_portfolio

if __name__ == "__main__":
    # Build a dictionary of DataFrames for multiple tickers using the Data Collection Agent.
    tickers = ["SPY", "QQQ", "IWM"]  # Example ETF tickers.
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    portfolio_data = {}

    data_agent = DataCollectionAgent()
    for ticker in tickers:
        print(f"Collecting data for {ticker}...")
        data = data_agent.collect_data(ticker, start_date, end_date)
        portfolio_data[ticker] = data

    # Analyze the portfolio data.
    agent = AnalysisAgent()
    analyzed_portfolio = agent.analyze_portfolio(portfolio_data)
    for ticker, analyzed_data in analyzed_portfolio.items():
        print(f"\nAnalysis for {ticker}:")
        print(analyzed_data.tail())
