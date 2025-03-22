import pandas as pd
import numpy as np

class AnalysisAgent:
    def __init__(self):
        pass

    def calculate_moving_average(self, data, window=20):
        data['MA'] = data['Close'].rolling(window=window).mean()
        return data

    def calculate_RSI(self, data, period=14):
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    def calculate_MACD(self, data, short_window=12, long_window=26, signal_window=9):
        data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
        data['MACD'] = data['EMA_short'] - data['EMA_long']
        data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
        return data

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        data['BB_MA'] = data['Close'].rolling(window=window).mean()
        data['BB_STD'] = data['Close'].rolling(window=window).std()
        data['BB_upper'] = data['BB_MA'] + num_std * data['BB_STD']
        data['BB_lower'] = data['BB_MA'] - num_std * data['BB_STD']
        return data

    def analyze_data(self, data):
        """
        Perform full technical analysis on a single DataFrame.
        """
        data = self.calculate_moving_average(data)
        data = self.calculate_RSI(data)
        data = self.calculate_MACD(data)
        data = self.calculate_bollinger_bands(data)
        return data

    def correlation_analysis(self, portfolio_data):
        """
        Given a dictionary of DataFrames for multiple assets, compute a correlation matrix
        of their closing prices.
        """
        prices = {ticker: df['Close'] for ticker, df in portfolio_data.items()}
        price_df = pd.DataFrame(prices)
        return price_df.corr()

    def analyze_portfolio(self, portfolio_data):
        """
        Analyze a portfolio of assets by processing each asset’s DataFrame.
        Returns a dictionary with ticker keys and analyzed DataFrames.
        """
        analyzed = {}
        for ticker, df in portfolio_data.items():
            analyzed[ticker] = self.analyze_data(df)
        return analyzed

if __name__ == "__main__":
    # Test analysis on dummy data.
    import numpy as np
    dates = pd.date_range(start="2022-01-01", periods=50, freq='B')
    dummy_data = pd.DataFrame({
        'Open': np.random.randn(50).cumsum() + 100,
        'High': np.random.randn(50).cumsum() + 102,
        'Low': np.random.randn(50).cumsum() + 98,
        'Close': np.random.randn(50).cumsum() + 100,
        'Volume': np.random.randint(1000000, 2000000, size=50)
    }, index=dates)
    agent = AnalysisAgent()
    analyzed = agent.analyze_data(dummy_data)
    print("Analyzed Data:")
    print(analyzed.head())
    corr = agent.correlation_analysis({"SPY": dummy_data, "QQQ": dummy_data})
    print("\nCorrelation Matrix:")
    print(corr)
