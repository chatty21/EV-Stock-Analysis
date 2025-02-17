from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend (non-GUI)
import plotly.express as px
import pandas as pd
import requests
from dotenv import load_dotenv
import plotly.graph_objs as go
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import seaborn as sns
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Get the API key for News API
API_KEY = os.getenv('NEWS_API_KEY')  # Get the API key from the environment variable

# Initialize the Flask app
app = Flask(__name__)

# Define file paths for your data (CSV files)
data_files = {
    'Tesla': 'data/TSLA_historical_data.csv',
    'Lucid': 'data/LCID_historical_data.csv',
    'Li Auto': 'data/LI_historical_data.csv',
    'NIO': 'data/NIO_historical_data.csv',
    'Nikola': 'data/NKLA_historical_data.csv',
    'Rivian': 'data/RIVN_historical_data.csv',
    'XPeng': 'data/XPEV_historical_data.csv'
}

# Function to fetch the latest news related to a specific topic (e.g., Tesla)
def fetch_latest_news(query):
    try:
        api_key = API_KEY
        url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error for bad HTTP status codes
        
        news_data = response.json()
        articles = news_data.get('articles', [])
        
        news = []
        for article in articles[:10]:  # Take the first 10 articles
            news.append({
                'title': article['title'],
                'description': article['description'],
                'url': article['url'],
                'publishedAt': article['publishedAt'],
            })
        return news
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return None

# Add features like Moving Averages and RSI
def add_technical_indicators(df):
    # Moving Averages (20-day and 50-day)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # Relative Strength Index (RSI) - 14-day
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Fill NaN values
    df.fillna(method='ffill', inplace=True)
    
    return df

# Preprocessing function
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in DateTime format
    df.set_index('Date', inplace=True)  # Set the 'Date' column as index
    df = df[['Close', 'Volume']]  # Use only 'Close' and 'Volume'
    df.fillna(method='ffill', inplace=True)  # Handle missing values
    return df


# Prepare data for LSTM model
def prepare_lstm_data(df, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)
    
    X = []
    y = []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])  # Use previous 'time_steps' days to predict the next day's close
        y.append(scaled_data[i, 0])  # Actual close price for the next day
    
    X = np.array(X)
    y = np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshaping for LSTM input
    
    return X, y, scaler

# Train LSTM model
def train_lstm_model(df, time_steps=60, epochs=10, batch_size=32):
    X, y, scaler = prepare_lstm_data(df, time_steps)
    
    split = int(0.8 * len(X))  # Split data into training and testing
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

# Predict future stock prices
def predict_future_prices(model, data, scaler, days=13):
    last_data = data[['Close']].iloc[-13:].values  # Use last 60 days of data for predictions
    last_data_scaled = scaler.transform(last_data)  # Scale the last data point
    
    future_predictions = []
    
    for _ in range(days):
        prediction = model.predict(last_data_scaled.reshape(1, -1, 1))
        future_predictions.append(prediction[0, 0])
        
        last_data_scaled = np.append(last_data_scaled[1:], prediction, axis=0)
        last_data_scaled = last_data_scaled.reshape(-1, 1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions

# Route to Stock Prediction Page
# Route to Prediction Page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    selected_company = 'Tesla'  # Default company
    if request.method == 'POST':
        selected_company = request.form['company']
    
    # Read the data for the selected company
    data = pd.read_csv(data_files[selected_company])
    data = preprocess_data(data)
    
    # Filter data for the last 5 months
    latest_date = data.index[-1]  # Get the latest date in the data
    twelve_months_ago = latest_date - pd.DateOffset(months=12)  # Subtract 5 months
    filtered_data = data.loc[data.index >= twelve_months_ago]  # Filter data to only include the last 5 months
    
    # Train the LSTM model
    y_train, y_pred, model, scaler = train_lstm_model(filtered_data)
    
    fig = go.Figure()

   # Plot actual stock prices
    fig.add_trace(go.Scatter(x=filtered_data.index[-len(y_train):], y=y_train.flatten(), mode='lines', name='Actual Stock Price'))

   # Plot predicted stock prices
    fig.add_trace(go.Scatter(x=filtered_data.index[-len(y_pred):], y=y_pred.flatten(), mode='lines', name='Predicted Stock Price'))

    fig.update_layout(
    title=f'{selected_company} Stock Price Prediction',
    xaxis_title='Date',
    yaxis_title='Stock Price'
)

    plot_html = fig.to_html(full_html=False)

    
    # Predict future stock prices (for Feb-Dec 2025)
    future_dates = pd.date_range(start='2025-02-01', end='2025-07-31', freq='B')  # Business days for 6 months
    future_predictions = predict_future_prices(model, filtered_data, scaler, days=len(future_dates))
    
    
    # Plot the future predictions with Plotly
    future_fig = px.line(x=future_dates, y=future_predictions.flatten(),
                         labels={'value': 'Stock Price', 'Date': 'Date'},
                         title=f'Future {selected_company} Stock Price Prediction (Feb-Dec 2025)')
    future_plot_html = future_fig.to_html(full_html=False)
    
    # Now render the template with plot_html and future_plot_html passed correctly
    return render_template('prediction.html', plot=plot_html, future_plot=future_plot_html, company=selected_company, companies=data_files.keys())

# Route to Stock Performance Page
@app.route('/performance', methods=['GET', 'POST'])
def performance():
    selected_company = 'Tesla'  # Default company
    if request.method == 'POST':
        selected_company = request.form['company']
    
    # Read the data for the selected company
    data = pd.read_csv(data_files[selected_company])
    
    # Check the columns and print for debugging
    print(data.columns)  # This will print the column names for debugging
    
    # Ensure the 'Date' column is in datetime format
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
    else:
        # If 'Date' column is named differently, adjust the name here
        print("No 'Date' column found!")
    
    # Rename columns if needed
    if 'Close' in data.columns:
        data.rename(columns={'Close': 'Close'}, inplace=True)  # Ensure 'Close' is the correct column
    else:
        print("No 'Close' column found!")

    # Create the Plotly line plot
    fig = px.line(data, x='Date', y='Close', title=f'{selected_company} Stock Price Trends')
    
    # Convert the plot to HTML for rendering
    plot_html = fig.to_html(full_html=False)
    
    return render_template('performance.html', plot=plot_html, company=selected_company, companies=data_files.keys())


# Route to Correlation Analysis Page
@app.route('/correlation')
def correlation():
    # Load stock data for all companies
    stock_data = {}
    for company, file in data_files.items():
        stock_data[company] = pd.read_csv(file)[['Date', 'Close']]  # Only date and closing price

    # Merge data on 'Date'
    merged_data = stock_data['Tesla']
    for company in stock_data:
        if company != 'Tesla':
            merged_data = pd.merge(merged_data, stock_data[company], on='Date', suffixes=('', f'_{company}'))

    # Drop the 'Date' column and calculate the correlation matrix
    correlation_data = merged_data.drop(columns='Date')
    correlation_matrix = correlation_data.corr()

    # Plot the correlation matrix using seaborn and matplotlib
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of EV Stocks')
    plt.tight_layout()

    # Save the plot to an image file
    correlation_img_path = 'static/correlation_matrix.png'
    plt.savefig(correlation_img_path)
    plt.close()  # Close the plot to free memory

    # Summary of the correlation analysis
    summary = (
        "The correlation analysis reveals how the stock prices of various EV companies move in relation to each other. "
        "A correlation value close to +1 indicates that the stocks tend to move in the same direction, while a value close "
        "to -1 means that the stocks move in opposite directions. For instance, if Tesla's stock is highly correlated with "
        "Lucid's stock, this suggests that both companies may be affected similarly by market events or industry trends. "
        "The heatmap visually represents these relationships."
    )

    return render_template('correlation.html', correlation_matrix=correlation_matrix, 
                           correlation_img_path=correlation_img_path, summary=summary)

# Route to Market Events Page
@app.route('/events')
def events():
    # Fetch latest news related to each company
    companies = ['Tesla', 'Lucid', 'Li Auto', 'NIO', 'Nikola', 'Rivian', 'XPeng']
    news_data = {}

    # Fetch news for each company (top 10 news articles)
    for company in companies:
        company_news = fetch_latest_news(f"{company} stock")
        news_data[company] = company_news if company_news else f"No news found for {company}."

    return render_template('events.html', news_data=news_data)

@app.route('/statistics')
def statistics():
    stats_data = {}

    for company, file in data_files.items():
        # Load the stock data for each company
        stock_data = pd.read_csv(file)

        # Preprocess the data
        stock_data = preprocess_data(stock_data)

        # Filter data for the last 5 years
        latest_date = stock_data.index[-1]  # Get the latest date in the data
        five_years_ago = latest_date - pd.DateOffset(years=5)  # Subtract 5 years
        filtered_data = stock_data.loc[stock_data.index >= five_years_ago]  # Filter data to only include the last 5 years

        # Calculate basic statistics for the filtered data
        stats = filtered_data['Close'].describe()  # Describe gives statistics like mean, std, etc.

        # Store the statistics in a dictionary
        stats_data[company] = stats

    return render_template('statistics.html', stats_data=stats_data)


# Route to Market Comparison Page
@app.route('/comparison')
def comparison():
    # Compare stock performance of multiple companies
    companies = ['Tesla', 'Lucid', 'Li Auto', 'XPeng', 'Nikola', 'Rivian', 'NIO']
    
    # Load data for the companies
    data = {}
    for company in companies:
        df = pd.read_csv(data_files[company])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Close']]
        df = df.rename(columns={'Close': company})
        data[company] = df

    # Merge data for comparison
    merged_data = data['Tesla']
    for company in companies[1:]:
        merged_data = pd.merge(merged_data, data[company], on='Date')
    
    fig = px.line(merged_data, x='Date', y=companies, title='Stock Price Comparison')
    plot_html = fig.to_html(full_html=False)
    
    return render_template('comparison.html', plot=plot_html)

# Route to Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5002)  # Run on port 5002 or another port if needed
