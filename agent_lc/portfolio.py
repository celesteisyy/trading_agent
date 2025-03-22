import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import logging
import warnings
import os
warnings.filterwarnings('ignore')


# Set up logging
log_dir = os.path.join("agent_lc", "output")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingSystem")

class PortfolioManagerAgent:
    """
    Agent responsible for portfolio management, position sizing, and risk controls
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []
        logger.info(f"Portfolio Manager initialized with ${initial_capital:,.2f} capital")
    
    def calculate_position_size(self, ticker, price, signal_strength):
        """
        Calculate position size based on risk parameters
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        price : float
            Current price of the asset
        signal_strength : float
            Strength of the trading signal (-1 to 1)
            
        Returns:
        --------
        dict
            Position details including size
        """
        # 1. Determine available capital
        available_capital = self.current_capital
        
        # 2. Calculate maximum position size (% of portfolio)
        max_position_pct = 0.2  # Maximum 20% of portfolio per position
        
        # 3. Adjust based on signal strength
        position_pct = max_position_pct * abs(signal_strength)
        
        # 4. Calculate number of shares
        position_value = available_capital * position_pct
        num_shares = int(position_value / price)
        
        # 5. Calculate actual position value
        actual_position_value = num_shares * price
        
        position_details = {
            'ticker': ticker,
            'shares': num_shares,
            'price': price,
            'value': actual_position_value,
            'portfolio_pct': actual_position_value / self.current_capital
        }
        
        logger.info(f"Calculated position size for {ticker}: {num_shares} shares (${actual_position_value:,.2f}, {position_details['portfolio_pct']:.2%} of portfolio)")
        
        return position_details
    
    def execute_trade(self, decision):
        """
        Execute a trade and update portfolio
        
        Parameters:
        -----------
        decision : dict
            Trade decision details
            
        Returns:
        --------
        dict
            Updated decision with execution details
        """
        action = decision['action']
        ticker = decision['ticker']
        date = decision['date']
        price = decision['price']
        
        updated_decision = decision.copy()
        
        if action == 'BUY':
            # Calculate position size
            signal_strength = decision.get('signal_strength', 0.5)
            position = self.calculate_position_size(ticker, price, signal_strength)
            
            # Check if we have enough capital
            if position['value'] > self.current_capital:
                logger.warning(f"Insufficient capital to execute BUY for {ticker}")
                updated_decision['action'] = 'REJECTED'
                updated_decision['reason'] = 'INSUFFICIENT_CAPITAL'
                return updated_decision
            
            # Update portfolio
            self.positions[ticker] = position
            self.current_capital -= position['value']
            
            updated_decision['shares'] = position['shares']
            updated_decision['value'] = position['value']
            logger.info(f"Executed BUY for {ticker}: {position['shares']} shares at ${price:.2f} (${position['value']:,.2f})")
            
        elif action == 'SELL':
            if ticker not in self.positions:
                logger.warning(f"Attempted to SELL {ticker} but no position exists")
                updated_decision['action'] = 'REJECTED'
                updated_decision['reason'] = 'NO_POSITION'
                return updated_decision
            
            # Calculate sale proceeds
            position = self.positions[ticker]
            sale_value = position['shares'] * price
            profit_loss = sale_value - position['value']
            
            # Update portfolio
            self.current_capital += sale_value
            del self.positions[ticker]
            
            updated_decision['shares'] = position['shares']
            updated_decision['value'] = sale_value
            updated_decision['profit_loss'] = profit_loss
            updated_decision['profit_pct'] = profit_loss / position['value'] * 100
            
            logger.info(f"Executed SELL for {ticker}: {position['shares']} shares at ${price:.2f} (${sale_value:,.2f}, P&L: ${profit_loss:,.2f}, {updated_decision['profit_pct']:.2f}%)")
        
        # Record trade
        self.trade_history.append(updated_decision)
        
        # Record portfolio state
        portfolio_value = self.calculate_portfolio_value(date)
        self.portfolio_history.append({
            'date': date,
            'cash': self.current_capital,
            'positions_value': portfolio_value - self.current_capital,
            'total_value': portfolio_value
        })
        
        return updated_decision
    
    def calculate_portfolio_value(self, current_date):
        """
        Calculate total portfolio value
        
        Parameters:
        -----------
        current_date : datetime
            Current date
            
        Returns:
        --------
        float
            Total portfolio value
        """
        # Cash component
        total_value = self.current_capital
        
        # Add value of all positions
        for ticker, position in self.positions.items():
            total_value += position['value']
        
        return total_value
    
    def get_portfolio_metrics(self):
        """
        Calculate portfolio performance metrics
        
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        if not self.portfolio_history:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
            }
        
        # Extract portfolio values
        values = [float(entry['total_value']) for entry in self.portfolio_history]
        
        # Calculate returns
        returns = pd.Series(values).pct_change().dropna()
        
        # Total return
        total_return = values[-1] - self.initial_capital
        total_return_pct = total_return / self.initial_capital * 100
        
        # Sharpe ratio (annualized, assuming daily data)
        mean_return = returns.mean()
        std_return = returns.std()
        risk_free_rate = 0.03 / 252  # Assuming 3% annual risk-free rate
        sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Maximum drawdown
        peak = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - peak) / peak
        max_drawdown = float(drawdown.min() * 100)  # Convert to percentage and ensure it's a scalar
        
        # Win rate
        profitable_trades = sum(1 for trade in self.trade_history 
                              if trade.get('action') == 'SELL' and 
                              float(trade.get('profit_loss', 0)) > 0)
        total_closed_trades = sum(1 for trade in self.trade_history 
                                if trade.get('action') == 'SELL')
        win_rate = profitable_trades / total_closed_trades * 100 if total_closed_trades > 0 else 0
        
        metrics = {
            'total_return': float(total_return),  # Ensure scalar values
            'total_return_pct': float(total_return_pct),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
        }
        
        logger.info(f"Portfolio Metrics: Return: {metrics['total_return_pct']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f}, Max DD: {metrics['max_drawdown']:.2f}%, Win Rate: {metrics['win_rate']:.2f}%")
        
        return metrics

