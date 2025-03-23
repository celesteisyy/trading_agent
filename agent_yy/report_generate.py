import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv, find_dotenv
import openai
import os

# Initialization: load environment variables and API key.
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

def generate_summary_report(data, additional_info=""):
    """
    Generate a summary report using OpenAI's LLM based on recent technical analysis.
    `data` is expected to be a pandas DataFrame; we take the last 5 rows for context.
    `additional_info` can include risk parameters or other context.
    """
    report_data = data.tail(5).to_csv(index=True)
    prompt = (
        "You are a trading strategy analyst. Based on the following recent technical indicator data (in CSV format), "
        "generate a concise summary report that includes your interpretation of the trends and your recommendations. "
        "Data:\n" + report_data +
        "\n" + additional_info +
        "\nPlease generate a report summary."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a trading strategy analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
        )
        report = response.choices[0].message.content.strip()
        return report
    except Exception as e:
        print("LLM report generation failed:", e)
        return "Report generation failed."

def plot_drawdown(portfolio_values):
    """
    Plot the portfolio drawdown over time.
    Drawdown is computed as the percentage difference from the running maximum.
    """
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(drawdown.index, drawdown.values, color='blue', label='Drawdown (%)')
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    where=(drawdown.values < 0),
                    color='red', alpha=0.3)

    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_trade_pnl_distribution(trade_pnl_series):
    """
    Plot the distribution of trade P&L (in percentage), with a KDE overlay and a vertical line at 0.
    trade_pnl_series: A pandas Series representing trade profit/loss in percentage terms.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    # Use seaborn histplot to display the histogram with a KDE overlay
    sns.histplot(trade_pnl_series, bins=20, kde=True, color='skyblue', ax=ax)
    # Draw a vertical line at x=0
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)

    ax.set_title("Trade P&L Distribution")
    ax.set_xlabel("Profit/Loss (%)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_portfolio_value(portfolio_values):
    """
    Plot the portfolio value over time.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(portfolio_values.index, portfolio_values.values, color='blue')
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value ($)")
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_portfolio_with_trades(portfolio_values, trades=None):
    """
    Plot the portfolio value over time, overlaying BUY/SELL signals with green/red markers.
    trades: a DataFrame with at least ['Date', 'action'] or ['Signal'] where
            'Date' is datetime-like and 'action'/'Signal' are either 'BUY' or 'SELL'.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(portfolio_values.index, portfolio_values.values, color='blue', label='Portfolio Value ($)')

    if trades is not None and not trades.empty:
        # Ensure the 'Date' column is in datetime format.
        trades['Date'] = pd.to_datetime(trades['Date'])
        buy_trades = trades[(trades['action'] == 'BUY') | (trades['Signal'] == 'BUY')]
        sell_trades = trades[(trades['action'] == 'SELL') | (trades['Signal'] == 'SELL')]

        # Overlay BUY/SELL markers on the portfolio value curve.
        if not buy_trades.empty:
            ax.scatter(buy_trades['Date'],
                       portfolio_values.loc[buy_trades['Date']],
                       marker='^', color='green', s=100, label='BUY')
        if not sell_trades.empty:
            ax.scatter(sell_trades['Date'],
                       portfolio_values.loc[sell_trades['Date']],
                       marker='v', color='red', s=100, label='SELL')
        ax.legend()

    ax.set_title("Portfolio Performance with Trades")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value ($)")
    ax.grid(True)
    fig.tight_layout()
    return fig
