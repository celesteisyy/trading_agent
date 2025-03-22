#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 15:49:06 2025

@author: wodewenjianjia
"""

from tradesystem import TradingSystem
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

def main():
    # Initialize the trading system
    system = TradingSystem(initial_capital=100000)
    
    # Define ETFs to trade
    etfs = ['SPY', 'QQQ', 'IWM']
    
    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
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