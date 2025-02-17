import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# Load and preprocess each stock dataset
def preprocess_data(df):
    # Ensure the 'Date' column is correctly set as the index
    df['Date'] = pd.to_datetime(df['Date'])  # Use the 'Date' column as DateTime
    df.set_index('Date', inplace=True)  # Set the 'Date' column as the index
    
    # We'll use 'Close' and 'Volume' columns for predictions
    df = df[['Close', 'Volume']]  # Only use the 'Close' and 'Volume' columns
    
    # Handle any missing values by filling forward (if any)
    df.fillna(method='ffill', inplace=True)
    
    return df

# Load the stock data from the uploaded files
nkla_data = pd.read_csv('data/NKLA_historical_data.csv')
lcid_data = pd.read_csv('data/LCID_historical_data.csv')
rivn_data = pd.read_csv('data/RIVN_historical_data.csv')
li_data = pd.read_csv('data/LI_historical_data.csv')
xpev_data = pd.read_csv('data/XPEV_historical_data.csv')
nio_data = pd.read_csv('data/NIO_historical_data.csv')
tsla_data = pd.read_csv('data/TSLA_historical_data.csv')

# Preprocess the data for each stock
nkla_processed = preprocess_data(nkla_data)
lcid_processed = preprocess_data(lcid_data)
rivn_processed = preprocess_data(rivn_data)
li_processed = preprocess_data(li_data)
xpev_processed = preprocess_data(xpev_data)
nio_processed = preprocess_data(nio_data)
tsla_processed = preprocess_data(tsla_data)

# Function to prepare the data for LSTM
def prepare_lstm_data(df, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)  # Scale only the 'Close' price
    
    X = []
    y = []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])  # Use previous 'time_steps' days to predict the next day's close
        y.append(scaled_data[i, 0])  # Actual close price for the next day
    
    X = np.array(X)
    y = np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Function to train and predict with LSTM
def train_lstm_model(df, time_steps=60, epochs=10, batch_size=32):
    X, y, scaler = prepare_lstm_data(df, time_steps)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Predict using the model
    y_pred = model.predict(X_train)
    
    y_pred = scaler.inverse_transform(y_pred)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    
    return y_train, y_pred, model, scaler

# Function to predict future prices for the next n days
def predict_future_prices(model, data, scaler, days=60):
    # Get the last available data point (only 'close' value)
    last_data = data[['Close']].iloc[-60:].values  # Only take the 'close' price for predictions
    
    # Scale the last data point
    last_data_scaled = scaler.transform(last_data)  # Scale the 'close' values
    
    # Prepare the future prediction
    future_predictions = []
    
    for _ in range(days):
        # Predict the next day's price
        prediction = model.predict(last_data_scaled.reshape(1, -1, 1))
        future_predictions.append(prediction[0, 0])
        
        # Update the last_data with the new predicted value (append and shift)
        last_data_scaled = np.append(last_data_scaled[1:], prediction, axis=0)
        last_data_scaled = last_data_scaled.reshape(-1, 1)  # Ensure the correct shape for next prediction
    
    # Inverse scale the predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions

# Dictionary of processed data for each stock
stocks_data = {
    'NKLA': nkla_processed,
    'LCID': lcid_processed,
    'RIVN': rivn_processed,
    'LI': li_processed,
    'XPEV': xpev_processed,
    'NIO': nio_processed,
    'TSLA': tsla_processed
}

predictions = {}
for stock, data in stocks_data.items():
    y_train, y_pred, model, scaler = train_lstm_model(data)
    predictions[stock] = {'y_train': y_train, 'y_pred': y_pred, 'model': model, 'scaler': scaler}

# Define the future dates for prediction (Feb-Mar 2025)
future_dates = pd.date_range(start='2025-02-01', end='2025-12-31', freq='B')  # Business days only

# Plotting the results for each stock with 'date' on the x-axis in separate graphs
for stock, result in predictions.items():
    plt.figure(figsize=(14, 7))
    
    # Get the dates from the stock data (align dates with y_train and y_pred)
    actual_dates = stocks_data[stock].iloc[len(stocks_data[stock]) - len(result['y_train']):].index
    
    # Plotting actual vs predicted prices
    plt.plot(actual_dates, result['y_train'], label=f'Actual {stock} Stock Price')
    plt.plot(actual_dates, result['y_pred'], label=f'Predicted {stock} Stock Price')
    
    # Predict future stock prices (Feb-Mar 2025)
    future_predictions = predict_future_prices(result['model'], stocks_data[stock], result['scaler'], days=len(future_dates))
    
    # Plotting future predicted prices (Feb-Mar 2025)
    plt.plot(future_dates, future_predictions, label=f'Predicted {stock} Stock Prices (Feb-Dec 2025)', linestyle='dashed', color='red')
    
    # Title and labels
    plt.title(f'Actual vs Predicted {stock} Stock Prices (From Listing to Present) and Future Prediction (Feb-Mar 2025)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45)
    
    # Ensure everything fits well in the plot
    plt.tight_layout()
    
    # Show the plot
    plt.show()