import pandas as pd
import numpy as np

class PortfolioManagementAgent:
    def __init__(self):
        # Initialize portfolio with starting cash and positions.
        self.cash = 100000.0  # Example initial cash.
        self.positions = {}   # Dictionary to hold positions for multiple tickers, e.g., {ticker: number of shares}.
        # List to store portfolio value history over trading days.
        self.portfolio_value_history = []

    def adjust_position(self, ticker, signal, price, quantity=100):
        """
        Adjust the portfolio position based on the trading signal for the given ticker:
          - BUY: Increase position and decrease cash.
          - SELL: Decrease position and increase cash.
          - HOLD: No change.
        Returns the updated positions and cash.
        """
        if signal == "BUY":
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            self.cash -= price * quantity
        elif signal == "SELL":
            current_qty = self.positions.get(ticker, 0)
            if current_qty >= quantity:
                self.positions[ticker] -= quantity
                self.cash += price * quantity
            else:
                # If insufficient shares, sell all available.
                self.cash += price * current_qty
                self.positions[ticker] = 0
        # HOLD signal does nothing.
        return self.positions, self.cash

    def execute_trades(self, trade_data, quantity=100):
        """
        Execute trades based on the trade_data provided by the strategy development agent.
        trade_data: a dictionary with ticker as key and a tuple (signal, price) as value.
        For each ticker, adjust the portfolio position according to the given signal and price.
        """
        for ticker, (signal, price) in trade_data.items():
            self.adjust_position(ticker, signal, price, quantity)

    def compute_portfolio_value(self, latest_prices):
        """
        Compute the total portfolio value given a dictionary of latest prices per ticker.
        latest_prices: A dictionary with structure {ticker: price}.
        Returns the total portfolio value (cash + positions valued at the latest prices).
        """
        total_value = self.cash
        for ticker, shares in self.positions.items():
            total_value += shares * latest_prices.get(ticker, 0)
        return total_value

    def update_portfolio_history(self, portfolio_value):
        """
        Append the latest portfolio value to the portfolio value history list.
        """
        self.portfolio_value_history.append(portfolio_value)

    def compute_performance_metrics(self):
        """
        Compute performance metrics based on the portfolio value history:
          - Cumulative Return: Total return over the period.
          - Maximum Drawdown: Maximum peak-to-trough decline.
          - Volatility: Standard deviation of daily returns.
          - Average Daily Return: Mean of daily returns.
        Returns a dictionary with these performance indicators.
        """
        if len(self.portfolio_value_history) < 2:
            return {}
        # Convert the history list to a pandas Series.
        values = pd.Series(self.portfolio_value_history)
        # Calculate daily returns.
        daily_returns = values.pct_change().dropna()
        cumulative_return = (values.iloc[-1] / values.iloc[0]) - 1
        # Calculate maximum drawdown.
        running_max = values.cummax()
        drawdowns = (values - running_max) / running_max
        max_drawdown = drawdowns.min()
        volatility = daily_returns.std()
        metrics = {
            "cumulative_return": cumulative_return,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "average_daily_return": daily_returns.mean()
        }
        return metrics

    def monitor_portfolio(self):
        """
        Output monitoring criteria and performance indicators based on the tracked portfolio value history.
        """
        metrics = self.compute_performance_metrics()
        if not metrics:
            print("Not enough data to compute performance metrics.")
            return
        print("Portfolio Monitoring Indicators:")
        print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Volatility (Std. Dev. of Daily Returns): {metrics['volatility']:.2%}")
        print(f"Average Daily Return: {metrics['average_daily_return']:.2%}")

if __name__ == "__main__":
    # Self-contained test for the portfolio management agent.
    agent = PortfolioManagementAgent()
    
    # Simulate trades for a single ticker "SPY" with sample signals.
    # Example trade data for a day: {ticker: (signal, price)}
    trade_data_day1 = {"SPY": ("BUY", 300)}
    agent.execute_trades(trade_data_day1)
    
    # Simulate updating portfolio value.
    latest_prices = {"SPY": 305}
    port_value = agent.compute_portfolio_value(latest_prices)
    agent.update_portfolio_history(port_value)
    
    # Simulate another trading day.
    trade_data_day2 = {"SPY": ("HOLD", 305)}
    agent.execute_trades(trade_data_day2)
    latest_prices = {"SPY": 310}
    port_value = agent.compute_portfolio_value(latest_prices)
    agent.update_portfolio_history(port_value)
    
    # Simulate a SELL signal on a subsequent day.
    trade_data_day3 = {"SPY": ("SELL", 310)}
    agent.execute_trades(trade_data_day3)
    latest_prices = {"SPY": 300}
    port_value = agent.compute_portfolio_value(latest_prices)
    agent.update_portfolio_history(port_value)
    
    # Output final portfolio status and performance metrics.
    print("\nFinal Portfolio Status:")
    print(f"Cash: {agent.cash:.2f}")
    print(f"Positions: {agent.positions}")
    final_value = agent.compute_portfolio_value(latest_prices)
    print(f"Final Portfolio Value: {final_value:.2f}")
    agent.monitor_portfolio()
