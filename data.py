import yfinance as yf
import os
import pandas as pd

# List of stock symbols for EV stocks
stocks = ['TSLA', 'NIO', 'XPEV', 'LI', 'NKLA', 'RIVN', 'LCID']

# Define the directory where the data should be saved
data_directory = '/Users/chaitanya/Desktop/EV STOCK/EV STOCKS ANALYSIS BASED SENTIMENTS/data'

# Create the 'data' directory if it doesn't exist
os.makedirs(data_directory, exist_ok=True)

# Function to fetch historical stock data
def fetch_stock_data(stock_symbol, start_date='2010-01-01', end_date='2025-01-31'):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data['Symbol'] = stock_symbol  # Add stock symbol to the data for identification
    return stock_data

# Fetch data for each stock and save it in the specified folder
for stock in stocks:
    stock_data = fetch_stock_data(stock)
    stock_data.to_csv(f"{data_directory}/{stock}_historical_data.csv")

    # Print a message after saving each stock's data
    print(f"Saved data for {stock} in {data_directory}/{stock}_historical_data.csv")

# Example: Display data for Tesla (TSLA)
print(f"Data for TSLA:\n{pd.read_csv(f'{data_directory}/TSLA_historical_data.csv').head()}")
