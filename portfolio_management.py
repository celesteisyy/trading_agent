class PortfolioManagementAgent:
    def __init__(self):
        # Initialize portfolio with starting cash and positions.
        self.cash = 100000  # Example initial cash.
        self.positions = {}  # Dictionary to hold positions for multiple tickers, e.g., {ticker: number of shares}.

    def adjust_position(self, ticker, signal, price, quantity=100):
        """
        Adjust the portfolio position based on the trading signal for the given ticker:
          - BUY: Increase position and decrease cash.
          - SELL: Decrease position and increase cash.
          - HOLD: No change.
        Update the portfolio value after the trade.
        """
        if signal == "BUY":
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            self.cash -= price * quantity
        elif signal == "SELL":
            current_qty = self.positions.get(ticker, 0)
            if current_qty >= quantity:
                self.positions[ticker] -= quantity
                self.cash += price * quantity
            else:
                # If insufficient shares, sell all available.
                self.cash += price * current_qty
                self.positions[ticker] = 0

        # Portfolio value is the sum of cash and all positions valued at the latest prices.
        return self.positions, self.cash

    def compute_portfolio_value(self, latest_prices):
        """
        Compute the total portfolio value given a dictionary of latest prices per ticker.
        latest_prices: dict with structure {ticker: price}
        """
        total_value = self.cash
        for ticker, shares in self.positions.items():
            total_value += shares * latest_prices.get(ticker, 0)
        return total_value

if __name__ == "__main__":
    # Test the portfolio management agent.
    agent = PortfolioManagementAgent()
    # Simulate a BUY for SPY
    pos, cash = agent.adjust_position("SPY", "BUY", 300)
    print(f"Positions: {pos}, Cash: {cash}")
