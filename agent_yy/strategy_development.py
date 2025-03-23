import pandas as pd

class StrategyDevelopmentAgent:
    def __init__(self):
        # Define risk management parameters, e.g., stop loss and take profit thresholds.
        self.stop_loss_threshold = 0.05   # 5%
        self.take_profit_threshold = 0.10 # 10%
    

    def compute_entry_price(self, analyzed_data):
        """
        Dynamically compute the entry price.
        Example: Use Bollinger Band lower band if available; otherwise, fallback to 98% of current price.
        Returns the entry price as a float.
        """
        last_row = analyzed_data.iloc[-1]
        # Check if 'BB_lower' column exists in the analyzed data
        if 'BB_lower' in analyzed_data.columns:
            bb_lower = last_row['BB_lower']
            # If bb_lower is a Series, use its first element; otherwise, use it directly.
            return float(bb_lower.iloc[0]) if hasattr(bb_lower, 'iloc') else float(bb_lower)
        else:
            # Fallback: return 98% of the current closing price.
            return float(last_row['Close']) * 0.98


    def generate_trade_signal(self, analyzed_data):
        """
        Generate a trade signal based on technical indicators.
        Steps:
        1. Extract the last row of the analyzed DataFrame.
        2. Convert RSI, MACD, Signal_Line, and current price to float using .iloc[0] if necessary.
        3. Dynamically compute the entry price.
        4. Determine the trading signal based on indicator thresholds.
        5. Apply risk management: override HOLD if the price deviates significantly.
        Returns:
        signal: The trading signal ("BUY", "SELL", or "HOLD").
        current_price: The current market price (float).
        """
        last_row = analyzed_data.iloc[-1]
        
        def to_float(val):
            return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
        
        rsi = to_float(last_row['RSI'])
        macd = to_float(last_row['MACD'])
        signal_line = to_float(last_row['Signal_Line'])
        current_price = to_float(last_row['Close'])
        
        # Compute dynamic entry price.
        entry_price = self.compute_entry_price(analyzed_data)
        
        if rsi < 30 and macd > signal_line:
            signal = "BUY"
        elif rsi > 70 and macd < signal_line:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Risk management: adjust signal if price deviates significantly from the entry price.
        if signal == "HOLD":
            if current_price <= entry_price * (1 - self.stop_loss_threshold):
                signal = "SELL"
            elif current_price >= entry_price * (1 + self.take_profit_threshold):
                signal = "SELL"
        
        return signal, current_price



    def generate_trade_instructions(self, analyzed_portfolio, trade_date):
        """
        Given an analyzed portfolio (a dict mapping tickers to analyzed DataFrames)
        and a trade_date, return a dictionary mapping each ticker to a tuple (signal, price).
        """
        instructions = {}
        for ticker, df in analyzed_portfolio.items():
            if trade_date in df.index:
                # Generate trade signal based on data up to the trade_date.
                subset = df.loc[:trade_date]
                signal, price = self.generate_trade_signal(subset)
                instructions[ticker] = (signal, price)
        return instructions

if __name__ == "__main__":
    # Test strategy on dummy analyzed data.
    import numpy as np
    dates = pd.date_range(start="2022-01-01", periods=50, freq='B')
    dummy_data = pd.DataFrame({
        'RSI': np.random.uniform(20, 80, 50),
        'MACD': np.random.randn(50),
        'Signal_Line': np.random.randn(50),
        'Close': np.random.randn(50).cumsum() + 100
    }, index=dates)
    agent = StrategyDevelopmentAgent()
    signal, price = agent.generate_trade_signal(dummy_data)
    print("Trade Signal:", signal, "at price:", price)
    analyzed_portfolio = {"SPY": dummy_data}
    trade_date = dummy_data.index[-1]
    instructions = agent.generate_trade_instructions(analyzed_portfolio, trade_date)
    print("Trade Instructions on", trade_date.date(), ":", instructions)
