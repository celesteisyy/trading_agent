import matplotlib.pyplot as plt
import seaborn as sns

# Import agent classes from separate files
from datacollect import DataCollectionAgent
from analysis import AnalysisAgent
from strategy import StrategyAgent
from portfolio import PortfolioManagerAgent

# Import other required libraries
import pandas as pd
import numpy as np
import datetime
import logging
import os

# Create output directory if it does not exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging configuration
log_path = os.path.join(output_dir, "trading_system.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingSystem")


class TradingSystem:
    """
    Main trading system that orchestrates all agents including data collection,
    analysis, strategy generation, and portfolio management.
    """

    def __init__(self, initial_capital=100000):
        # Initialize all system agents
        self.data_agent = DataCollectionAgent()
        self.analysis_agent = AnalysisAgent()
        self.strategy_agent = StrategyAgent(min_holding_period=5, max_trades_per_week=1)
        self.portfolio_agent = PortfolioManagerAgent(initial_capital=initial_capital)
        self.tickers = []
        self.start_date = None
        self.end_date = None
        self.current_date = None
        logger.info("Trading System initialized")

    def setup(self, tickers, start_date, end_date=None):
        """
        Set up the trading system with the given tickers and date range.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols to trade.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format; defaults to today if not provided.
        """
        self.tickers = tickers

        # Convert string dates to pd.Timestamp for consistent handling
        if isinstance(start_date, str):
            self.start_date = pd.Timestamp(start_date)
        else:
            self.start_date = pd.Timestamp(start_date)

        if end_date is None:
            self.end_date = pd.Timestamp(datetime.datetime.now().strftime('%Y-%m-%d'))
        else:
            if isinstance(end_date, str):
                self.end_date = pd.Timestamp(end_date)
            else:
                self.end_date = pd.Timestamp(end_date)

        # Convert timestamps to strings for yfinance download
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = self.end_date.strftime('%Y-%m-%d')

        # Fetch market data for the given tickers and date range
        self.data_agent.fetch_data(tickers, start_str, end_str)
        self.data_agent.handle_missing_data()
        self.data_agent.remove_outliers()

        # Calculate technical indicators and generate trading signals for each ticker
        for ticker in tickers:
            if ticker in self.data_agent.data:
                if not self.data_agent.data[ticker].empty:
                    self.analysis_agent.calculate_indicators(self.data_agent.data[ticker], ticker)
                    self.analysis_agent.generate_signals_very_simple(ticker)
                else:
                    logger.warning(f"Empty data for {ticker}, skipping indicator calculation")

        logger.info(f"Trading system setup complete for {len(tickers)} tickers from {start_str} to {end_str}")

    def run_backtest(self):
        """
        Run the backtest for the configured system and execute trades based on generated signals.
        
        Returns:
        --------
        dict
            A dictionary containing backtest metrics, trade history, and portfolio history.
        """
        logger.info("Starting backtest")

        # Collect all trading days from the available ticker data
        all_dates = []
        for ticker in self.tickers:
            if ticker in self.data_agent.data and not self.data_agent.data[ticker].empty:
                dates = self.data_agent.data[ticker].index.tolist()
                all_dates.extend(dates)

        if not all_dates:
            logger.error("No trading dates found in any ticker data")
            return {
                'metrics': self.portfolio_agent.get_portfolio_metrics(),
                'trade_history': [],
                'portfolio_history': []
            }

        # Convert all dates to pandas Timestamps for consistency
        all_dates = [pd.Timestamp(d) for d in all_dates]

        # Create a unique sorted list of trading days
        trading_days = sorted(list(set(all_dates)))

        # Filter trading days to be within the specified date range
        filtered_days = [day for day in trading_days if (day >= self.start_date) and (day <= self.end_date)]
        trading_days = filtered_days

        if not trading_days:
            logger.error("No trading days found in the specified date range")
            return {
                'metrics': self.portfolio_agent.get_portfolio_metrics(),
                'trade_history': [],
                'portfolio_history': []
            }

        logger.info(f"Found {len(trading_days)} trading days in date range")

        # Store indicator data with calculated signals for each ticker
        indicator_data = {}
        for ticker in self.tickers:
            if ticker in self.analysis_agent.indicators:
                indicator_data[ticker] = self.analysis_agent.indicators[ticker]
                logger.info(f"Columns in {ticker} indicator data: {list(indicator_data[ticker].columns)}")
                if 'Signal' in indicator_data[ticker].columns:
                    logger.info(f"First few signal values for {ticker}: {indicator_data[ticker]['Signal'].head().tolist()}")

        # Loop through each trading day to execute trades based on signals
        for day in trading_days:
            self.current_date = day

            for ticker in self.tickers:
                if ticker not in indicator_data:
                    continue

                signals_df = indicator_data[ticker]
                if day not in signals_df.index:
                    continue

                if 'Signal' not in signals_df.columns:
                    logger.warning(f"No 'Signal' column found for {ticker} on {day}")
                    continue

                # Generate trade decision for the day
                decision = self.strategy_agent.generate_trade_decisions(signals_df, ticker, day)

                # Execute trade if decision action is BUY or SELL
                if decision['action'] in ['BUY', 'SELL']:
                    self.strategy_agent.execute_trade(decision)
                    self.portfolio_agent.execute_trade(decision)

        # Calculate final portfolio metrics
        metrics = self.portfolio_agent.get_portfolio_metrics()

        logger.info("Backtest completed")
        logger.info(f"Final portfolio value: ${self.portfolio_agent.calculate_portfolio_value(self.end_date):,.2f}")
        logger.info(f"Total return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Maximum drawdown: {metrics['max_drawdown']:.2f}%")
        logger.info(f"Win rate: {metrics['win_rate']:.2f}%")

        return {
            'metrics': metrics,
            'trade_history': self.portfolio_agent.trade_history,
            'portfolio_history': self.portfolio_agent.portfolio_history
        }

    def generate_performance_report(self, results):
        """
        Generate a performance report based on backtest results. This includes creating visualizations
        such as portfolio value, drawdown, and trade performance charts, and saving them to the output folder.
        
        Parameters:
        -----------
        results : dict
            The backtest results containing metrics, trade history, and portfolio history.
        
        Returns:
        --------
        dict
            A dictionary containing report data and visualizations.
        """
        logger.info("Generating performance report")

        metrics = results['metrics']
        trade_history = results['trade_history']
        portfolio_history = results['portfolio_history']

        # Convert portfolio history to DataFrame for visualization
        portfolio_df = pd.DataFrame(portfolio_history)

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not portfolio_df.empty and 'date' in portfolio_df.columns:
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)

            if 'total_value' in portfolio_df.columns:
                plt.figure(figsize=(12, 8))
                plt.plot(portfolio_df.index, portfolio_df['total_value'])
                plt.title('Portfolio Value Over Time')
                plt.xlabel('Date')
                plt.ylabel('Value ($)')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'portfolio_value.png'))

                if len(trade_history) > 0:
                    plt.figure(figsize=(12, 8))
                    plt.plot(portfolio_df.index, portfolio_df['total_value'])
                    trades_df = pd.DataFrame(trade_history)
                    if not trades_df.empty and 'date' in trades_df.columns:
                        trades_df['date'] = pd.to_datetime(trades_df['date'])
                        buys = trades_df[trades_df['action'] == 'BUY']
                        sells = trades_df[trades_df['action'] == 'SELL']

                        for _, trade in buys.iterrows():
                            if 'value' in trade:
                                plt.scatter(trade['date'], trade['value'], color='green', marker='^', s=100)

                        for _, trade in sells.iterrows():
                            if 'value' in trade:
                                plt.scatter(trade['date'], trade['value'], color='red', marker='v', s=100)
                    plt.title('Portfolio Performance with Trades')
                    plt.xlabel('Date')
                    plt.ylabel('Value ($)')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'portfolio_with_trades.png'))

                portfolio_df['previous_peak'] = portfolio_df['total_value'].cummax()
                portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['previous_peak']) / portfolio_df['previous_peak'] * 100

                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_df.index, portfolio_df['drawdown'])
                plt.fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, color='red', alpha=0.3)
                plt.title('Portfolio Drawdown')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'drawdown.png'))
        else:
            logger.warning("Empty portfolio history or missing required columns for visualization")
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            values = [self.portfolio_agent.initial_capital] * len(dates)

            plt.figure(figsize=(12, 8))
            plt.plot(dates, values)
            plt.title('Portfolio Value (No Trades)')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'portfolio_value.png'))

        trades_df = pd.DataFrame(trade_history)
        if not trades_df.empty and 'profit_pct' in trades_df.columns:
            profit_trades = trades_df[trades_df['action'] == 'SELL']
            if not profit_trades.empty and 'profit_pct' in profit_trades.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(profit_trades['profit_pct'], bins=20, kde=True)
                plt.axvline(x=0, color='red', linestyle='--')
                plt.title('Trade P&L Distribution')
                plt.xlabel('Profit/Loss (%)')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'pnl_distribution.png'))

        logger.info("Performance report generated")

        return {
            'metrics': metrics,
            'portfolio_df': portfolio_df if not portfolio_df.empty else pd.DataFrame(),
            'trades_df': trades_df if not trades_df.empty else pd.DataFrame()
        }

    def export_results(self, results_dir='./results'):
        """
        Export the results (trade history, portfolio history, and performance metrics) to CSV files.
        
        Parameters:
        -----------
        results_dir : str
            Directory where the results CSV files will be saved.
        """
        # Use output directory for results export
        results_dir = output_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        trades_df = pd.DataFrame(self.portfolio_agent.trade_history)
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(results_dir, "trade_history.csv"), index=False)
            logger.info(f"Trade history exported to {os.path.join(results_dir, 'trade_history.csv')}")

        portfolio_df = pd.DataFrame(self.portfolio_agent.portfolio_history)
        if not portfolio_df.empty:
            portfolio_df.to_csv(os.path.join(results_dir, "portfolio_history.csv"), index=False)
            logger.info(f"Portfolio history exported to {os.path.join(results_dir, 'portfolio_history.csv')}")

        metrics = self.portfolio_agent.get_portfolio_metrics()
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(results_dir, "performance_metrics.csv"), index=False)
        logger.info(f"Performance metrics exported to {os.path.join(results_dir, 'performance_metrics.csv')}")

        logger.info(f"All results exported to {results_dir}")

    def create_dashboard(self):
        """
        Create an interactive dashboard to display system performance metrics and visualizations.
        
        Returns:
        --------
        Dashboard object
            The Dash app instance containing the dashboard.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import dash
            from dash import dcc, html
            import dash_bootstrap_components as dbc

            logger.info("Creating performance dashboard")

            # Convert portfolio history to DataFrame for visualization
            portfolio_df = pd.DataFrame(self.portfolio_agent.portfolio_history)

            if portfolio_df.empty:
                logger.warning("Empty portfolio history, dashboard will be limited")
                # Create a dummy DataFrame with initial capital for each day in the date range
                date_range = pd.date_range(start=self.start_date, end=self.end_date)
                values = [self.portfolio_agent.initial_capital] * len(date_range)
                portfolio_df = pd.DataFrame({
                    'date': date_range,
                    'total_value': values
                })
                portfolio_df.set_index('date', inplace=True)
            else:
                portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
                portfolio_df.set_index('date', inplace=True)

            trades_df = pd.DataFrame(self.portfolio_agent.trade_history)
            if not trades_df.empty:
                trades_df['date'] = pd.to_datetime(trades_df['date'])

            # Get portfolio metrics for display
            metrics = self.portfolio_agent.get_portfolio_metrics()

            # Create Dash app with Bootstrap styling
            app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

            # Create Plotly figures for the dashboard
            fig1 = go.Figure()
            fig2 = go.Figure()
            fig3 = go.Figure()

            # Portfolio Value Chart
            fig1.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['total_value'],
                mode='lines',
                name='Portfolio Value'
            ))

            # Add BUY/SELL markers if trade data is available
            if not trades_df.empty:
                buys = trades_df[trades_df['action'] == 'BUY']
                sells = trades_df[trades_df['action'] == 'SELL']

                if not buys.empty:
                    fig1.add_trace(go.Scatter(
                        x=buys['date'],
                        y=buys['value'],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        name='Buy'
                    ))

                if not sells.empty:
                    fig1.add_trace(go.Scatter(
                        x=sells['date'],
                        y=sells['value'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name='Sell'
                    ))

            fig1.update_layout(
                title='Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )

            # Drawdown Chart
            if 'total_value' in portfolio_df.columns:
                portfolio_df['previous_peak'] = portfolio_df['total_value'].cummax()
                portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['previous_peak']) / portfolio_df['previous_peak'] * 100

                fig2.add_trace(go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['drawdown'],
                    fill='tozeroy',
                    mode='lines',
                    line=dict(color='red'),
                    name='Drawdown'
                ))

                fig2.update_layout(
                    title='Portfolio Drawdown',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    yaxis=dict(tickformat='.2f')
                )

            # Trade P&L Distribution Chart
            if not trades_df.empty and 'profit_pct' in trades_df.columns:
                profit_trades = trades_df[trades_df['action'] == 'SELL']['profit_pct']
                if not profit_trades.empty:
                    fig3 = go.Figure(data=[go.Histogram(
                        x=profit_trades,
                        nbinsx=20,
                        marker_color='blue',
                        name='P&L Distribution'
                    )])
                    fig3.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                    fig3.update_layout(
                        title='Trade P&L Distribution',
                        xaxis_title='Profit/Loss (%)',
                        yaxis_title='Frequency'
                    )

            # Dashboard layout using Dash Bootstrap Components
            app.layout = dbc.Container([
                html.H1("Trading System Performance Dashboard", className="text-center my-4"),

                # Metrics cards
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Total Return"),
                        dbc.CardBody(html.H4(f"{metrics['total_return_pct']:.2f}%"))
                    ]), width=3),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Sharpe Ratio"),
                        dbc.CardBody(html.H4(f"{metrics['sharpe_ratio']:.2f}"))
                    ]), width=3),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Max Drawdown"),
                        dbc.CardBody(html.H4(f"{metrics['max_drawdown']:.2f}%"))
                    ]), width=3),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Win Rate"),
                        dbc.CardBody(html.H4(f"{metrics['win_rate']:.2f}%"))
                    ]), width=3)
                ], className="mb-4"),

                # Charts section
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig1), width=12)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig2), width=6),
                    dbc.Col(dcc.Graph(figure=fig3), width=6)
                ], className="mb-4"),

                # Trade history table
                html.H3("Recent Trades", className="mt-4"),
                html.Div([
                    dash.dash_table.DataTable(
                        id='trade-table',
                        columns=[
                            {"name": "Date", "id": "date"},
                            {"name": "Ticker", "id": "ticker"},
                            {"name": "Action", "id": "action"},
                            {"name": "Price", "id": "price"},
                            {"name": "Shares", "id": "shares"},
                            {"name": "Value", "id": "value"},
                            {"name": "Profit/Loss %", "id": "profit_pct"}
                        ],
                        data=trades_df.sort_values('date', ascending=False).head(10).to_dict('records') if not trades_df.empty else [],
                        style_cell={'textAlign': 'center'},
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{action} = "BUY"'},
                                'backgroundColor': 'rgba(0, 255, 0, 0.2)'
                            },
                            {
                                'if': {'filter_query': '{action} = "SELL"'},
                                'backgroundColor': 'rgba(255, 0, 0, 0.2)'
                            }
                        ]
                    )
                ])
            ], fluid=True)

            logger.info("Dashboard created successfully")

            return app

        except ImportError as e:
            logger.error(f"Could not create dashboard due to missing dependencies: {str(e)}")
            logger.info("Install required packages with: pip install dash dash-bootstrap-components plotly")
            return None