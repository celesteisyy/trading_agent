import pandas as pd
from datetime import datetime

# Import the agents
from data_collection import DataCollectionAgent
from analysis import AnalysisAgent
from strategy_development import StrategyDevelopmentAgent
from portfolio_management import PortfolioManagementAgent

def main():
    # Define a list of ETFs for the portfolio.
    tickers = ["SPY", "QQQ", "IWM"]  # Example ETF tickers.
    start_date = "2022-01-01"
    end_date = "2022-12-31"

    # Initialize agents.
    data_agent = DataCollectionAgent()
    analysis_agent_obj = AnalysisAgent()
    strategy_agent = StrategyDevelopmentAgent()
    portfolio_agent = PortfolioManagementAgent()

    # Dictionary to store processed data for each ticker.
    portfolio_data = {}

    # 1. Data Collection, Analysis, and Signal Generation for each ticker.
    for ticker in tickers:
        data = data_agent.collect_data(ticker, start_date, end_date)
        analyzed_data = analysis_agent_obj.analyze(data)
        signal_data = strategy_agent.generate_signal(analyzed_data)
        portfolio_data[ticker] = signal_data
        print(f"Data processing completed for {ticker}.")

    # 2. Simulate weekly trading for each ticker.
    # Here we assume that trading days are aligned across ETFs.
    # We use the index from the first ticker as the reference trading days.
    sample_ticker = tickers[0]
    trade_dates = portfolio_data[sample_ticker].index[::5]  # Every 5th day as trading day.

    # Dictionary to hold the latest prices for portfolio value computation.
    latest_prices = {}

    for trade_date in trade_dates:
        print(f"\nTrading Date: {trade_date.date()}")
        for ticker in tickers:
            # Ensure the ticker's data contains the current trading date.
            if trade_date in portfolio_data[ticker].index:
                day_data = portfolio_data[ticker].loc[trade_date]
                signal = day_data['Signal']
                price = day_data['Close']
                # Update latest price for the ticker.
                latest_prices[ticker] = price
                positions, cash = portfolio_agent.adjust_position(ticker, signal, price)
                print(f"Ticker: {ticker}, Signal: {signal}, Price: {price:.2f}")
        # Compute and display portfolio value after each trading day.
        port_value = portfolio_agent.compute_portfolio_value(latest_prices)
        print(f"Updated Cash: {portfolio_agent.cash:.2f}, Portfolio Value: {port_value:.2f}")

    # 3. Final Portfolio Status
    final_value = portfolio_agent.compute_portfolio_value(latest_prices)
    print("\nFinal Portfolio Status:")
    print(f"Cash: {portfolio_agent.cash:.2f}")
    print(f"Positions: {portfolio_agent.positions}")
    print(f"Final Portfolio Value: {final_value:.2f}")

if __name__ == "__main__":
    main()
