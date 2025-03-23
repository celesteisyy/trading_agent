import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import openai
from data_collection import DataCollectionAgent
from analysis import AnalysisAgent
from strategy_development import StrategyDevelopmentAgent
from report_generate import (generate_summary_report, 
                             plot_drawdown, 
                             plot_pnl_distribution, 
                             plot_portfolio_value, 
                             plot_portfolio_with_trades)
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches

# Initialization: load environment variables and API key.
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

def main():
    print("Project: ETF and Major Asset Classes Multi-Agent Trading System")
    print("Objective: Data collection, analysis, strategy construction, and report generation.\n")
    
    # Define output directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 0: Fetch available ETFs.
    data_agent = DataCollectionAgent()
    available_etfs = data_agent.get_available_etfs()
    if not available_etfs:
        available_etfs = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VNQ"]
    
    # Save available ETFs as CSV.
    etfs_df = pd.DataFrame(available_etfs, columns=['Ticker'])
    etfs_csv_file = os.path.join(output_dir, "available_etfs.csv")
    etfs_df.to_csv(etfs_csv_file, index=False)
    print("Available ETFs CSV saved at:", etfs_csv_file)
    
    # Ask user to select ETFs.
    tickers_input = input("Please check the available ETFs CSV file and enter the tickers you want to select (comma-separated): ")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip().upper() in available_etfs]
    if not tickers:
        print("No valid tickers selected. Defaulting to SPY, QQQ, IWM.")
        tickers = ["SPY", "QQQ", "IWM"]
    
    # Set time window (last 5 years).
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    print(f"Time Window: {start_date_str} to {end_date_str}")
    
    # Initialize Analysis and Strategy Development agents.
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
    sample_trade_date = analyzed_portfolio[tickers[0]].index[10]
    trade_instructions = strategy_agent.generate_trade_instructions(analyzed_portfolio, sample_trade_date)
    print("\nTrade Instructions on", sample_trade_date.date(), ":", trade_instructions)
    
    # Step 4: Report Generation.
    try:
        report_text = generate_summary_report(analyzed_portfolio[tickers[0]],
                                              additional_info="Risk-Free Rate: " + str(risk_free_rate))
    except Exception as e:
        report_text = f"LLM report generation failed: {e}"
        print(report_text)
    
    # For demonstration, assume the portfolio value series is derived from the "Close" column of the first asset.
    # In practice, you would compute your actual portfolio value over time.
    portfolio_value_series = analyzed_portfolio[tickers[0]]["Close"]
    
    # Also, assume you have trade data available; if not, set trades_data to None.
    trades_data = None  # Replace with actual DataFrame of trades if available.
    
    # Generate four separate plots.
    fig_drawdown = plot_drawdown(portfolio_value_series)
    fig_pnl = plot_pnl_distribution(portfolio_value_series)
    fig_value = plot_portfolio_value(portfolio_value_series)
    fig_with_trades = plot_portfolio_with_trades(portfolio_value_series, trades=trades_data)
    
    # Save each plot to file.
    plot_files = {}
    if fig_drawdown:
        plot_files['Drawdown'] = os.path.join(output_dir, "drawdown.png")
        fig_drawdown.savefig(plot_files['Drawdown'])
        plt.close(fig_drawdown)
    if fig_pnl:
        plot_files['PnL'] = os.path.join(output_dir, "pnl.png")
        fig_pnl.savefig(plot_files['PnL'])
        plt.close(fig_pnl)
    if fig_value:
        plot_files['Portfolio Value'] = os.path.join(output_dir, "portfolio_value.png")
        fig_value.savefig(plot_files['Portfolio Value'])
        plt.close(fig_value)
    if fig_with_trades:
        plot_files['Portfolio with Trades'] = os.path.join(output_dir, "portfolio_with_trades.png")
        fig_with_trades.savefig(plot_files['Portfolio with Trades'])
        plt.close(fig_with_trades)
    
    for name, path in plot_files.items():
        print(f"{name} plot saved at: {path}")
    
    # Create a DOCX report that includes the LLM report text and all generated images.
    doc = Document()
    doc.add_heading("Strategy Report", 0)
    doc.add_paragraph(report_text)
    doc.add_heading("Generated Plots", level=1)
    for name, path in plot_files.items():
        if os.path.exists(path):
            doc.add_heading(name, level=2)
            doc.add_picture(path, width=Inches(6))
    report_docx_file = os.path.join(output_dir, "report.docx")
    doc.save(report_docx_file)
    print("\nGenerated Strategy Report saved at:", report_docx_file)
    
if __name__ == "__main__":
    main()
