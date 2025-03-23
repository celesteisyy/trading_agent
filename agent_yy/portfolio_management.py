import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from strategy_development import TradingStrategyAgent
from data_collection import DataCollectionAgent

class PortfolioManagerAgent:
    """
    Agent responsible for portfolio management, position sizing, and risk controls.
    It integrates the trading strategy from StrategyAgent and retrieves needed market data
    and risk free rate from DataCollectionAgent.
    """
    def __init__(self, initial_capital=100000, strategy_agent=None, data_agent=None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # Record current positions.
        self.trade_history = []  # Store executed trades.
        self.portfolio_history = []  # History of portfolio values.
        self.trades = []  # Additional trade records.
        # Use the provided strategy agent or instantiate a default StrategyAgent.
        self.strategy_agent = strategy_agent if strategy_agent else TradingStrategyAgent()
        # Use the provided data agent or instantiate a default DataCollectionAgent.
        self.data_agent = data_agent if data_agent else DataCollectionAgent()
    
    def calculate_position_size(self, ticker, price, signal_strength):
        """
        Calculate position size based on available capital and signal strength.
        
        Parameters:
            ticker (str): Ticker symbol.
            price (float): Execution price.
            signal_strength (float): Factor to adjust position size.
        
        Returns:
            dict: Position details including number of shares and position value.
        """
        if price <= 0:
            return {
                'ticker': ticker,
                'shares': 0,
                'price': price,
                'value': 0,
                'portfolio_pct': 0
            }
            
        available_capital = self.current_capital
        max_position_pct = 0.2 
        position_pct = max_position_pct * abs(signal_strength)
        position_value = available_capital * position_pct
        num_shares = int(position_value / price)
        actual_position_value = num_shares * price
        position_details = {
            'ticker': ticker,
            'shares': num_shares,
            'price': price,
            'value': actual_position_value,
            'portfolio_pct': actual_position_value / self.current_capital if self.current_capital > 0 else 0
        }
        return position_details
    
    def generate_trade_decision(self, ticker, analyzed_data, trade_date):
        """
        Generate a trade decision using the StrategyAgent.
        
        Parameters:
            ticker (str): Ticker symbol.
            analyzed_data (pd.DataFrame): DataFrame containing technical indicators.
            trade_date (Timestamp): Date to evaluate the trade signal.
            
        Returns:
            dict: Trade decision details with keys such as 'ticker', 'action', 'price', and 'date'.
        """
        # Use all data up to trade_date and get the decision from the strategy agent.
        decision = self.strategy_agent.generate_trade_decision(analyzed_data, ticker, trade_date)
        return decision
    
    def execute_trade(self, decision):
        """
        Execute a trade based on the provided decision.
        
        Parameters:
            decision (dict): Trade decision details (must include 'ticker', 'action', and 'price').
            
        Returns:
            dict: Formatted decision with additional execution details.
        """
        if not decision:
            return {"action": "INVALID", "reason": "Empty decision"}
            
        ticker = decision.get('ticker')
        action = decision.get('action')
        price = decision.get('price')
        trade_date = decision.get('date', pd.Timestamp.now())
        
        # Check if price is valid.
        if price is None or price <= 0:
            return {
                'ticker': ticker,
                'action': 'INVALID',
                'reason': 'Invalid price',
                'date': pd.Timestamp.now()
            }
        
        # Build the formatted decision dictionary.
        formatted_decision = {
            'ticker': ticker,
            'action': action,
            'price': price,
            'date': trade_date
        }
        
        if action == 'BUY':
            # [BUY logic...]
            pass
        elif action == 'SELL':
            # [SELL logic...]
            pass
        elif action == 'HOLD':
            formatted_decision['value'] = 0
        
        self.trades.append(formatted_decision)
        self.trade_history.append(formatted_decision)
        
        return formatted_decision

    
    def adjust_position(self, ticker, signal, price, quantity=100):
        """
        Adjust the portfolio position based on a direct trading signal.
        
        Parameters:
            ticker (str): Ticker symbol.
            signal (str): Trading signal ("BUY" or "SELL").
            price (float): Execution price.
            quantity (int): Number of shares to adjust.
        
        Returns:
            tuple: Updated positions dictionary and current capital.
        """
        if price <= 0 or quantity <= 0:
            return self.positions, self.current_capital
            
        if signal == "BUY":
            # Check if enough capital exists.
            if price * quantity > self.current_capital:
                return self.positions, self.current_capital
                
            current_shares = self.positions[ticker]['shares'] if ticker in self.positions else 0
            self.positions[ticker] = {
                'entry_price': price if current_shares == 0 else 
                               (self.positions[ticker]['entry_price'] * current_shares + price * quantity) / 
                               (current_shares + quantity),
                'entry_date': self.positions[ticker]['entry_date'] if ticker in self.positions else pd.Timestamp.now(),
                'shares': current_shares + quantity
            }
            self.current_capital -= price * quantity
        elif signal == "SELL":
            if ticker in self.positions:
                current_shares = self.positions[ticker]['shares']
                if current_shares >= quantity:
                    self.positions[ticker]['shares'] = current_shares - quantity
                    self.current_capital += price * quantity
                    if self.positions[ticker]['shares'] == 0:
                        del self.positions[ticker]
                else:
                    self.current_capital += price * current_shares
                    del self.positions[ticker]
        
        return self.positions, self.current_capital
    
    def compute_portfolio_value(self, latest_prices):
        """
        Compute the total portfolio value as the sum of current cash and market value of positions.
        
        Parameters:
            latest_prices (dict): Dictionary mapping tickers to their latest prices.
        
        Returns:
            float: Total portfolio value.
        """
        total_value = self.current_capital
        for ticker, position_info in self.positions.items():
            if ticker in latest_prices and latest_prices[ticker] > 0:
                total_value += position_info['shares'] * latest_prices[ticker]
        return total_value
    
    def update_portfolio_history(self, portfolio_value):
        """
        Append the current portfolio value and cash to the portfolio history.
        
        Parameters:
            portfolio_value (float): Total portfolio value.
        """
        self.portfolio_history.append({
            'date': pd.Timestamp.now(),
            'total_value': portfolio_value,
            'cash': self.current_capital
        })
    
    def get_portfolio_metrics(self):
        """
        Compute and return performance metrics based on portfolio history.
        Uses the risk free rate fetched from the DataCollectionAgent.
        
        Returns:
            dict: Performance metrics including cumulative return, Sharpe ratio,
                  maximum drawdown, and win rate.
        """
        if len(self.portfolio_history) < 2:
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
        
        # Retrieve the daily risk-free rate using DataCollectionAgent.
        risk_free_daily = self.data_agent.get_risk_free_rate()
        if risk_free_daily is None or risk_free_daily <= 0:
            # Fallback to a default daily risk free rate (e.g., 0.03 annualized)
            risk_free_daily = 0.03 / 252
        
        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe_ratio = (mean_return - risk_free_daily) / std_return * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
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
        Print key portfolio performance metrics.
        """
        metrics = self.get_portfolio_metrics()
        print("Portfolio Monitoring Indicators:")
        print(f"Cumulative Return: {metrics['total_return_pct']:.2f}%")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        
    def plot_portfolio_performance(self, save_path=None):
        """
        Plot portfolio performance over time.
        
        Parameters:
            save_path (str, optional): Path to save the plot. If None, display the plot.
        """
        if len(self.portfolio_history) < 2:
            print("Not enough data to plot portfolio performance.")
            return
            
        dates = [entry['date'] for entry in self.portfolio_history]
        values = [entry['total_value'] for entry in self.portfolio_history]
        cash = [entry['cash'] for entry in self.portfolio_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, label='Total Portfolio Value', linewidth=2)
        plt.plot(dates, cash, label='Cash', linestyle='--')
        plt.axhline(y=self.initial_capital, color='r', linestyle='-', label='Initial Capital')
        
        plt.title('Portfolio Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

if __name__ == "__main__":
    # Use DataCollectionAgent to gather real market data for a given ticker.
    ticker = "SPY"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    
    data_agent = DataCollectionAgent()
    market_data = data_agent.collect_data(ticker)
    
    if market_data.empty:
        print("Market data collection failed. Please check the ticker or date range.")
    else:
        # Ensure the necessary columns for strategy signals are present.
        # For demonstration, if technical indicators or signals are missing, fill with dummy values.
        if 'Signal' not in market_data.columns:
            market_data['Signal'] = np.random.uniform(-1, 1, len(market_data))
        for col in ['RSI', 'MACD', 'MACD_Signal']:
            if col not in market_data.columns:
                market_data[col] = np.random.uniform(20, 80, len(market_data)) if col == 'RSI' else np.random.randn(len(market_data))
        if 'BB_Lower' not in market_data.columns:
            market_data['BB_Lower'] = market_data['Close'] * 0.95
        
        # Instantiate the portfolio manager with the integrated strategy and data agents.
        portfolio_agent = PortfolioManagerAgent(initial_capital=100000, data_agent=data_agent)
        
        # Day 1: Generate a trade decision and execute the trade.
        trade_date_day1 = market_data.index[10]
        decision_day1 = portfolio_agent.generate_trade_decision(ticker, market_data, trade_date_day1)
        execution_result_day1 = portfolio_agent.execute_trade(decision_day1)
        print(f"Day 1 Decision: {decision_day1}")
        print(f"Day 1 Execution: {execution_result_day1}")
        
        latest_prices = {ticker: decision_day1.get("price", 0)}
        port_value = portfolio_agent.compute_portfolio_value(latest_prices)
        portfolio_agent.update_portfolio_history(port_value)
        
        # Day 2: Generate a new decision based on updated data.
        trade_date_day2 = market_data.index[20]
        decision_day2 = portfolio_agent.generate_trade_decision(ticker, market_data, trade_date_day2)
        execution_result_day2 = portfolio_agent.execute_trade(decision_day2)
        print(f"\nDay 2 Decision: {decision_day2}")
        print(f"Day 2 Execution: {execution_result_day2}")
        
        latest_prices = {ticker: decision_day2.get("price", 0)}
        port_value = portfolio_agent.compute_portfolio_value(latest_prices)
        portfolio_agent.update_portfolio_history(port_value)
        
        # Day 3: Generate a decision and force a SELL action for testing.
        trade_date_day3 = market_data.index[30]
        decision_day3 = portfolio_agent.generate_trade_decision(ticker, market_data, trade_date_day3)
        if decision_day3.get("action") != "SELL":
            decision_day3["action"] = "SELL"
        execution_result_day3 = portfolio_agent.execute_trade(decision_day3)
        print(f"\nDay 3 Decision: {decision_day3}")
        print(f"Day 3 Execution: {execution_result_day3}")
        
        latest_prices = {ticker: decision_day3.get("price", 0)}
        port_value = portfolio_agent.compute_portfolio_value(latest_prices)
        portfolio_agent.update_portfolio_history(port_value)
        
        # Display final portfolio status and performance metrics.
        print("\nFinal Portfolio Status:")
        print(f"Cash: ${portfolio_agent.current_capital:.2f}")
        print(f"Positions: {portfolio_agent.positions}")
        final_value = portfolio_agent.compute_portfolio_value(latest_prices)
        print(f"Final Portfolio Value: ${final_value:.2f}")
        portfolio_agent.monitor_portfolio()
        
        # Plot portfolio performance.
        try:
            portfolio_agent.plot_portfolio_performance()
        except Exception as e:
            print(f"Could not plot portfolio performance: {e}")
