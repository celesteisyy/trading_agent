from tradesystem import TradingSystem
from report_generate import ReportAgent
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import os
import pandas as pd

# Set up logging
log_dir = os.path.join("output")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "trading_system.log")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

def main():
    # Initialize the trading system with initial capital
    system = TradingSystem(initial_capital=100000)
    
    # Retrieve available ETFs from the data agent and save them to a CSV for reference
    available_etfs = system.data_agent.get_available_etfs()
    if available_etfs:
        csv_path = os.path.join("output", "available_etfs.csv")
        pd.DataFrame(available_etfs, columns=["Ticker"]).to_csv(csv_path, index=False)
    
    # Prompt the user to input ticker names (comma-separated, e.g., SPY,QQQ,IWM)
    user_input = input("Enter the ticker names of the ETFs to trade (comma-separated, e.g., SPY,QQQ,IWM): ")
    try:
        # Process the input string into a list of tickers
        selected_etfs = [ticker.strip().upper() for ticker in user_input.split(",") if ticker.strip()]
        if not selected_etfs:
            raise ValueError("No valid tickers entered.")
    except Exception as e:
        print("Invalid input. Using default ETFs ['SPY', 'QQQ', 'IWM'].")
        selected_etfs = ['SPY', 'QQQ', 'IWM']
    
    # Set the date range for the backtest (past 5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - relativedelta(years=5)).strftime('%Y-%m-%d')
    
    # Set up the trading system with the user-selected ETFs and date range
    system.setup(selected_etfs, start_date, end_date)
    
    # Run the backtest and obtain results
    results = system.run_backtest()
    
    # Generate performance report data from backtest results.
    # This report_data is a dict with keys 'metrics', 'portfolio_df', and 'trades_df'
    report_data = system.generate_performance_report(results)
    
    # Generate final report analysis using ReportAgent
    report_agent = ReportAgent(system.data_agent)
    # Retrieve additional financial info (e.g., risk-free rate) to include in the report
    risk_free_rate = system.data_agent.get_risk_free_rate()
    additional_info = f"Risk-free rate (10-Year Treasury): {risk_free_rate}" if risk_free_rate is not None else ""
    # Generate the final DOCX report using the performance report data
    final_report_path = report_agent.generate_final_report(report_data, system.portfolio_agent, additional_info)
    logger.info(f"Final report generated at: {final_report_path}")
    
    # Create and start the dashboard server to display interactive performance metrics
    dashboard = system.create_dashboard()
    if dashboard:
        print("Starting dashboard server...")
        print("Access the dashboard at http://127.0.0.1:8050/")
        print("Press Ctrl+C to stop the server")
        dashboard.run(debug=False)

if __name__ == "__main__":
    main()
