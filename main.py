from tradesystem import TradingSystem
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import os


# Set up logging
log_dir = os.path.join("output")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "trading_system.log")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")


def main():
    # Initialize the trading system
    system = TradingSystem(initial_capital=100000)
    
    # Define ETFs to trade
    etfs = ['SPY', 'QQQ', 'IWM']
    
    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - relativedelta(years=5)).strftime('%Y-%m-%d')
    
    # Set up trading system
    system.setup(etfs, start_date, end_date)
    
    # Run backtest
    results = system.run_backtest()
    
    # Generate reports
    system.generate_performance_report(results)
    
    # Create dashboard
    dashboard = system.create_dashboard()
    if dashboard:
        print("Starting dashboard server...")
        print("Access the dashboard at http://127.0.0.1:8050/")
        print("Press Ctrl+C to stop the server")
        dashboard.run(debug=False) 

if __name__ == "__main__":
    main()