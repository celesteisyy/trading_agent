import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import openai

def generate_summary_report(data, additional_info=""):
    """
    Generate a summary report using OpenAI’s LLM based on recent technical analysis.
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
            model="gpt-3.5-turbo",  # 或者 "gpt-4" 如果可用
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
    Generate a plot showing the portfolio drawdown.
    Drawdown is computed as the percentage difference between the portfolio value and its cumulative maximum.
    """
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max * 100
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(drawdown.index, drawdown.values, label="Drawdown (%)", color="red")
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_pnl_distribution(portfolio_values):
    """
    Generate a histogram showing the distribution of daily PnL (difference in portfolio value day-to-day).
    """
    pnl = portfolio_values.diff().dropna()
    fig, ax = plt.subplots(figsize=(12,6))
    ax.hist(pnl, bins=20, color="skyblue", edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", label="Zero PnL")
    ax.set_title("Daily PnL Distribution")
    ax.set_xlabel("Daily PnL")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_portfolio_value(portfolio_values):
    """
    Generate a plot of the portfolio value over time.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(portfolio_values.index, portfolio_values.values, label="Portfolio Value", color="blue")
    ax.set_title("Portfolio Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_portfolio_with_trades(portfolio_values, trades=None):
    """
    Generate a plot of portfolio value over time with trade markers.
    `trades` should be a DataFrame containing at least a 'Date' column and a 'Signal' column (e.g., BUY/SELL).
    """
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(portfolio_values.index, portfolio_values.values, label="Portfolio Value", color="blue")
    
    if trades is not None and not trades.empty:
        trades['Date'] = pd.to_datetime(trades['Date'])
        buy_trades = trades[trades['Signal'] == "BUY"]
        sell_trades = trades[trades['Signal'] == "SELL"]
        if not buy_trades.empty:
            ax.scatter(buy_trades['Date'], portfolio_values.loc[buy_trades['Date']],
                       marker="^", color="green", s=100, label="Buy")
        if not sell_trades.empty:
            ax.scatter(sell_trades['Date'], portfolio_values.loc[sell_trades['Date']],
                       marker="v", color="red", s=100, label="Sell")
        ax.legend()
    
    ax.set_title("Portfolio Value with Trades")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return fig
