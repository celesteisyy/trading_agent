# lc trading agent
## Overview
This project implements a backtesting trading system. It collects historical market data, processes it (handles missing values and replaces outliers with rolling medians), calculates technical indicators, generates trading signals, and manages a portfolio with trade execution. An optional interactive dashboard displays performance metrics.

## File Structure
- **main.py**  
  The entry point. It sets up the trading system, defines the date range and ETFs (e.g., SPY, QQQ, IWM), runs the backtest, and launches the dashboard.

- **datacollect.py**  
  Contains `DataCollectionAgent` for fetching market data via yfinance, handling missing data, and removing outliers using a rolling median. Logs are output to `agent_lc/output/trading_system.log`.

- **analysis.py**  
  Contains `AnalysisAgent` for calculating technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR) and generating trading signals.

- **strategy.py**  
  Contains `StrategyAgent` which applies strategy rules (e.g., minimum holding period, maximum trades per week) to generate and execute trade decisions. Logs output similarly.

- **portfolio.py**  
  Contains `PortfolioManagerAgent` that manages the portfolio, calculates position sizes, executes trades, and computes performance metrics like total return, Sharpe ratio, and max drawdown.

- **tradesystem.py**  
  Contains `TradingSystem` to orchestrate the data collection, analysis, strategy, and portfolio management modules, as well as report generation and dashboard creation.

## Setup & Usage
1. **Installation:**
   - Clone the repository.
   - Create and activate a virtual environment.
   - Install dependencies (e.g., pandas, numpy, matplotlib, seaborn, yfinance, dash, plotly).

2. **Running the Backtest:**
   Execute:
   ```bash
   python main.py
