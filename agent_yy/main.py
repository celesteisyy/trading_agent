import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import openai
from data_collection import DataCollectionAgent
from strategy_development import StrategyAgent
from portfolio_management import PortfolioManagerAgent
from report_generate import ReportAgent  # ReportAgent now includes generate_final_report()
from datetime import datetime
import matplotlib.pyplot as plt

# Initialization: Load environment variables and API key.
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

def main():
    print("Project: ETF and Major Asset Classes Multi-Agent Trading System")
    print("Objective: Full simulation using fetched data, strategy & portfolio management, and final report generation via report_generate.\n")
    
    # Define the output directory.
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 0: Retrieve available ETFs and save as a CSV for reference.
    data_agent = DataCollectionAgent()  # Time window is set within this agent.
    available_etfs = data_agent.get_available_etfs()
    if not available_etfs:
        available_etfs = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VNQ"]
    etfs_df = pd.DataFrame(available_etfs, columns=['Ticker'])
    etfs_csv_file = os.path.join(output_dir, "available_etfs.csv")
    etfs_df.to_csv(etfs_csv_file, index=False)
    print("Available ETFs CSV saved at:", etfs_csv_file)
    
    tickers_input = input("Please check the available ETFs CSV file and enter the tickers you want to select (comma-separated): ")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip().upper() in available_etfs]
    if not tickers:
        print("No valid tickers selected. Defaulting to SPY, QQQ, IWM.")
        tickers = ["SPY", "QQQ", "IWM"]
    
    # Step 1: Collect market data (past 5 years) for each selected ticker.
    portfolio_data = {}
    for ticker in tickers:
        print(f"\nCollecting data for {ticker}...")
        data = data_agent.collect_data(ticker)  # DataCollectionAgent uses its internal time window.
        portfolio_data[ticker] = data
        ratios = data_agent.get_financial_ratios(ticker)
        if ratios:
            print(f"Financial ratios for {ticker} retrieved successfully.")
    
    risk_free_rate = data_agent.get_risk_free_rate()
    if risk_free_rate is not None:
        print("\nLatest Risk-Free Rate (10-Year Treasury):", risk_free_rate)
    
    # Step 2: Use the fetched data as the "analyzed" portfolio.
    analyzed_portfolio = portfolio_data
    for ticker in tickers:
        print(f"Data collection completed for {ticker}.")
    
    # Step 3: Strategy development and portfolio management.
    strategy_agent = StrategyAgent()
    portfolio_agent = PortfolioManagerAgent()
    
    # Simulate trading on each trading day (for simplicity, using dates from the first ticker).
    trading_dates = analyzed_portfolio[tickers[0]].index
    for date in trading_dates:
        # Generate trade instructions for all tickers.
        trade_instructions = strategy_agent.generate_trade_decisions(analyzed_portfolio, date)
        for ticker, (signal, price) in trade_instructions.items():
            if signal != "HOLD":
                decision = {
                    'ticker': ticker,
                    'action': signal,
                    'price': price,
                    'date': date
                }
                portfolio_agent.execute_trade(decision)
        # Update portfolio value using the latest closing prices from all tickers.
        latest_prices = {}
        for ticker in tickers:
            if date in analyzed_portfolio[ticker].index:
                latest_prices[ticker] = analyzed_portfolio[ticker].loc[date, "Close"]
        port_val = portfolio_agent.compute_portfolio_value(latest_prices)
        portfolio_agent.update_portfolio_history(port_val)
    
    # Step 4: Call ReportAgent to generate the final report.
    # The ReportAgent's generate_final_report method (implemented in report_generate.py)
    # handles both plot generation and DOCX report creation.
    report_agent = ReportAgent(data_agent)
    additional_info = "Risk-Free Rate: " + str(risk_free_rate)
    report_agent.generate_final_report(analyzed_portfolio[tickers[0]], portfolio_agent, additional_info)
    
if __name__ == "__main__":
    main()
