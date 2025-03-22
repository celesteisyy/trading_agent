import pandas as pd
import matplotlib.pyplot as plt
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

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
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.5,
            n=1,
            stop=None,
        )
        report = response.choices[0].text.strip()
        return report
    except Exception as e:
        print("LLM report generation failed:", e)
        return "Report generation failed."

def plot_time_series(time_series, title="Time Series Chart", ylabel="Value"):
    """
    Generate and display a time series plot using matplotlib.
    """
    plt.figure()
    plt.plot(time_series.index, time_series.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test plotting with dummy time series data.
    import numpy as np
    dates = pd.date_range(start="2022-01-01", periods=30, freq='B')
    dummy_series = pd.Series(np.random.randn(30).cumsum(), index=dates)
    plot_time_series(dummy_series, title="Dummy Portfolio Value", ylabel="Value")
    
    # Test generating a dummy summary report.
    dummy_df = pd.DataFrame({
        "MA": np.random.randn(30),
        "RSI": np.random.uniform(20, 80, 30),
        "MACD": np.random.randn(30),
        "Signal": np.random.choice(["BUY", "SELL", "HOLD"], 30)
    }, index=dates)
    additional_info = "Risk-Free Rate: 4.24% and Market Premium: 5%"
    report = generate_summary_report(dummy_df, additional_info=additional_info)
    print("Generated Report:")
    print(report)
