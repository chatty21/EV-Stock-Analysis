import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define file paths for each company's stock data
data_files = {
    'Tesla': 'data/TSLA_historical_daily_adjusted.csv',
    'Lucid': 'data/LCID_historical_daily_adjusted.csv',
    'Li Auto': 'data/LI_historical_daily_adjusted.csv',
    'NIO': 'data/NIO_historical_daily_adjusted.csv',
    'Nikola': 'data/NKLA_historical_daily_adjusted.csv',
    'Rivian': 'data/RIVN_historical_daily_adjusted.csv',
    'XPeng': 'data/XPEV_historical_daily_adjusted.csv'
}

# Function to preprocess stock data for each company and train XGBoost
def train_xgboost_model(company, file_path):
    # Load the stock data
    data = pd.read_csv(file_path)

    # Convert the 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # Feature Engineering: Create lag features for stock price
    data['prev_close'] = data['4. close'].shift(1)  # Previous day's closing price
    data['prev_2_close'] = data['4. close'].shift(2)  # Closing price from two days ago

    # Drop missing values
    data.dropna(inplace=True)

    # Define features (X) and target (y)
    X = data[['prev_close', 'prev_2_close']]
    y = data['4. close']  # Target: Today's closing price

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create and train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=10)
    model.fit(X_train, y_train)

    # Predict stock prices on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance using Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for {company}: {mse}')

    # Predict the next 60 days (February-March 2025)
    last_date = data.index[-1]
    future_dates = pd.date_range(last_date, periods=60, freq='D')
    
    # Create future predictions (next 60 days)
    future_data = pd.DataFrame(index=future_dates)
    future_data['prev_close'] = data['4. close'].iloc[-1]  # Use last available value
    future_data['prev_2_close'] = data['4. close'].iloc[-2]  # Use the second to last available value
    
    future_pred = model.predict(future_data[['prev_close', 'prev_2_close']])

    # Plot actual vs predicted stock prices
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['4. close'], label='Actual', color='blue')
    plt.plot(future_dates, future_pred, label='Predicted', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{company} Stock Price Prediction (Feb-Mar 2025)')
    plt.legend()
    plt.show()

    return model, future_pred, future_dates

# Train models for each stock and predict for February-March 2025
models = {}
for company, file_path in data_files.items():
    model, future_pred, future_dates = train_xgboost_model(company, file_path)
    print(f"{company} - Future Predictions for Feb-Mar 2025:")
    print(future_dates)
    print(future_pred)
