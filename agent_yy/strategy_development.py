import pandas as pd

class StrategyDevelopmentAgent:
    def __init__(self):
        # Define risk management parameters, e.g., stop loss and take profit thresholds.
        self.stop_loss_threshold = 0.05   # 5%
        self.take_profit_threshold = 0.10 # 10%

    def generate_trade_signal(self, analyzed_data):
        """
        Given an analyzed DataFrame (with indicators), generate a trade signal.
        For example, use simple rules:
          - If RSI < 30 and MACD > Signal_Line, signal BUY.
          - If RSI > 70 and MACD < Signal_Line, signal SELL.
          - Otherwise, HOLD.
        Incorporate risk management (for demonstration, using a dummy entry price).
        """
        last_row = analyzed_data.iloc[-1]
        rsi = last_row['RSI']
        macd = last_row['MACD']
        signal_line = last_row['Signal_Line']
        current_price = last_row['Close']
        # For risk management, assume a dummy entry price.
        entry_price = current_price * 0.98

        if rsi < 30 and macd > signal_line:
            signal = "BUY"
        elif rsi > 70 and macd < signal_line:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Example risk management: if price has moved sufficiently, override to SELL.
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
