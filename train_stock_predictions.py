import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load Tesla data to check
tsla_data = pd.read_csv('data/TSLA_historical_data.csv')

# Check the first few rows to ensure it's in the correct format
print(tsla_data.head())
