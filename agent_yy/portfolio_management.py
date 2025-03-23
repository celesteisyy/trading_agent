import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class PortfolioManagerAgent:
    """
    Agent responsible for portfolio management, position sizing, and risk controls.
    """
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []
    
    def calculate_position_size(self, ticker, price, signal_strength):
        """
        Calculate position size based on available capital and signal strength.
        """
        available_capital = self.current_capital
        max_position_pct = 0.2  # Maximum 20% of portfolio per position
        position_pct = max_position_pct * abs(signal_strength)
        position_value = available_capital * position_pct
        num_shares = int(position_value / price)
        actual_position_value = num_shares * price
        position_details = {
            'ticker': ticker,
            'shares': num_shares,
            'price': price,
            'value': actual_position_value,
            'portfolio_pct': actual_position_value / self.current_capital
        }
        return position_details
    
    def execute_trade(self, decision):
        """
        Execute a trade based on the decision.

        Parameters:
        decision : dict
            Trade decision details (must include 'ticker', 'action', 'price', and 'date').

        Returns:
        dict
            Updated decision with execution details.
        """
        ticker = decision['ticker']
        action = decision['action']

        if action == 'BUY':
            # Record the buy trade by storing entry price and date.
            self.positions[ticker] = {
                'entry_price': decision['price'],
                'entry_date': decision['date'],
                'size': 1  # Example: fixed size; adjust as needed.
            }
        elif action == 'SELL':
            if ticker in self.positions:
                # Calculate profit percentage
                entry_price = self.positions[ticker]['entry_price']
                exit_price = decision['price']
                profit_pct = (exit_price - entry_price) / entry_price * 100
                decision['profit_pct'] = profit_pct
                # Remove the position after selling
                del self.positions[ticker]
            else:
                decision['action'] = 'INVALID'
        
        # Record the executed trade
        self.trades.append(decision)
        return decision

    
    def adjust_position(self, ticker, signal, price, quantity=100):
        """
        Adjust portfolio position based on the trading signal.
        """
        if signal == "BUY":
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            self.current_capital -= price * quantity
        elif signal == "SELL":
            current_qty = self.positions.get(ticker, 0)
            if current_qty >= quantity:
                self.positions[ticker] -= quantity
                self.current_capital += price * quantity
            else:
                self.current_capital += price * current_qty
                self.positions[ticker] = 0
        return self.positions, self.current_capital
    
    def compute_portfolio_value(self, latest_prices):
        """
        Compute total portfolio value: cash plus the value of positions.
        latest_prices: dict in the form {ticker: price}.
        """
        total_value = self.current_capital
        for ticker, shares in self.positions.items():
            total_value += shares * latest_prices.get(ticker, 0)
        return total_value
    
    def update_portfolio_history(self, portfolio_value):
        """
        Append the latest portfolio value to the history.
        """
        self.portfolio_history.append({
            'date': pd.Timestamp.now(),
            'total_value': portfolio_value,
            'cash': self.current_capital
        })
    
    def get_portfolio_metrics(self):
        """
        Compute performance metrics based on portfolio history.
        """
        if not self.portfolio_history:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
            }
        values = [float(entry['total_value']) for entry in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()
        total_return = values[-1] - self.initial_capital
        total_return_pct = total_return / self.initial_capital * 100
        mean_return = returns.mean()
        std_return = returns.std()
        risk_free_rate = 0.03 / 252  # Assume 3% annual risk-free rate
        sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252) if std_return > 0 else 0
        peak = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - peak) / peak
        max_drawdown = float(drawdown.min() * 100)
        profitable_trades = sum(1 for trade in self.trade_history 
                                if trade.get('action') == 'SELL' and float(trade.get('profit_loss', 0)) > 0)
        total_closed_trades = sum(1 for trade in self.trade_history if trade.get('action') == 'SELL')
        win_rate = profitable_trades / total_closed_trades * 100 if total_closed_trades > 0 else 0
        metrics = {
            'total_return': float(total_return),
            'total_return_pct': float(total_return_pct),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
        }
        return metrics
    
    def monitor_portfolio(self):
        """
        Print portfolio performance metrics.
        """
        metrics = self.get_portfolio_metrics()
        if not metrics:
            print("Not enough data to compute performance metrics.")
            return
        print("Portfolio Monitoring Indicators:")
        print(f"Cumulative Return: {metrics['total_return_pct']:.2f}%")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")

if __name__ == "__main__":
    # Test PortfolioManagerAgent
    agent = PortfolioManagerAgent()
    trade_data_day1 = {"SPY": ("BUY", 300)}
    agent.execute_trade(trade_data_day1)
    latest_prices = {"SPY": 305}
    port_value = agent.compute_portfolio_value(latest_prices)
    agent.update_portfolio_history(port_value)
    trade_data_day2 = {"SPY": ("HOLD", 305)}
    agent.execute_trade(trade_data_day2)
    latest_prices = {"SPY": 310}
    port_value = agent.compute_portfolio_value(latest_prices)
    agent.update_portfolio_history(port_value)
    trade_data_day3 = {"SPY": ("SELL", 310)}
    agent.execute_trade(trade_data_day3)
    latest_prices = {"SPY": 300}
    port_value = agent.compute_portfolio_value(latest_prices)
    agent.update_portfolio_history(port_value)
    print("\nFinal Portfolio Status:")
    print(f"Cash: {agent.current_capital:.2f}")
    print(f"Positions: {agent.positions}")
    final_value = agent.compute_portfolio_value(latest_prices)
    print(f"Final Portfolio Value: {final_value:.2f}")
    agent.monitor_portfolio()
