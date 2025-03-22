import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import openai
from data_collection import DataCollectionAgent
from analysis import AnalysisAgent
from strategy_development import StrategyDevelopmentAgent
from report_generate import generate_summary_report, plot_time_series
from datetime import datetime
from dateutil.relativedelta import relativedelta

# initialization
_ = load_dotenv(find_dotenv())
openai.api_key  = os.environ['OPENAI_API_KEY']

def main():
    print("Project: ETF and Major Asset Classes Multi-Agent Trading System")
    print("Objective: Implement data collection, analysis, strategy construction, and portfolio management.\n")
    
    # Initialize Data Collection Agent to fetch available ETFs.
    data_agent = DataCollectionAgent()
    available_etfs = data_agent.get_available_etfs()
    if not available_etfs:
        # Fallback default list if API call fails.
        available_etfs = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VNQ"]
    print("Available ETFs:", ", ".join(available_etfs))
    
    # Ask user to select ETFs from the available list.
    tickers_input = input("Please enter the tickers you want to select (comma-separated): ")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip().upper() in available_etfs]
    if not tickers:
        print("No valid tickers selected. Defaulting to SPY, QQQ, IWM.")
        tickers = ["SPY", "QQQ", "IWM"]
    
    # Set time window to the last 5 years.
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    print(f"Time Window: {start_date_str} to {end_date_str}")
    
    # Initialize remaining agents.
    analysis_agent_obj = AnalysisAgent()
    strategy_agent = StrategyDevelopmentAgent()
    
    # Step 1: Data Collection.
    portfolio_data = {}
    for ticker in tickers:
        print(f"\nCollecting data for {ticker}...")
        data = data_agent.collect_data(ticker, start_date_str, end_date_str)
        portfolio_data[ticker] = data
        
        ratios = data_agent.get_financial_ratios(ticker)
        if ratios:
            print(f"Financial ratios for {ticker} retrieved successfully.")
    
    risk_free_rate = data_agent.get_risk_free_rate()
    if risk_free_rate is not None:
        print("\nLatest Risk-Free Rate (10-Year Treasury):", risk_free_rate)
    
    # Step 2: Analysis.
    analyzed_portfolio = analysis_agent_obj.analyze_portfolio(portfolio_data)
    for ticker, df in analyzed_portfolio.items():
        print(f"Analysis completed for {ticker}.")
    
    # Step 3: Strategy Development.
    # For demonstration, choose a sample trade date from the first selected ETF.
    sample_trade_date = analyzed_portfolio[tickers[0]].index[10]
    trade_instructions = strategy_agent.generate_trade_instructions(analyzed_portfolio, sample_trade_date)
    print("\nTrade Instructions on", sample_trade_date.date(), ":", trade_instructions)
    
    # Step 4: Report Generation.
    report = generate_summary_report(analyzed_portfolio[tickers[0]],
                                     additional_info="Risk-Free Rate: " + str(risk_free_rate))
    print("\nGenerated Strategy Report:")
    print(report)
    
    # Plot a time series chart for the selected asset's closing prices.
    selected_asset_close = analyzed_portfolio[tickers[0]]["Close"]
    plot_time_series(selected_asset_close, title=f"{tickers[0]} Closing Prices", ylabel="Price")
    
if __name__ == "__main__":
    main()