import yfinance as yf
import pandas as pd

# List of stock symbols for EV stocks
stocks = ['TSLA', 'NIO', 'XPEV', 'LI', 'NKLA', 'RIVN', 'LCID']

# Function to fetch historical stock data
def fetch_stock_data(stock_symbol, start_date='2010-01-01', end_date='2025-01-31'):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data['Symbol'] = stock_symbol  # Add stock symbol to the data for identification
    return stock_data

# Fetch data for each stock
stocks_data = {}
for stock in stocks:
    stocks_data[stock] = fetch_stock_data(stock)

# Example: Display data for Tesla (TSLA)
print(stocks_data['TSLA'].head())

# Save the data to CSV if you want to persist it for later use
for stock, data in stocks_data.items():
    data.to_csv(f"{stock}_historical_data.csv")
