# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Function to download historical stock data from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to preprocess the stock data
def preprocess_data(stock_data):
    # Drop any rows with missing values
    stock_data.dropna(inplace=True)
    return stock_data

# Function to create features and target variable
def create_features_target(stock_data, target_column='Close', window_size=30):
    # Create lagged features
    for i in range(1, window_size+1):
        stock_data[f'lag_{i}'] = stock_data[target_column].shift(i)
    
    # Drop rows with missing values
    stock_data.dropna(inplace=True)
    
    # Features are lagged values
    X = stock_data[[f'lag_{i}' for i in range(1, window_size+1)]]
    
    # Target variable is the current close price
    y = stock_data[target_column]
    
    return X, y

# Function to train a linear regression model
def train_model(X_train, y_train):
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Main function
def main():
    # Define parameters
    tickers = ['AAPL', 'MSFT']  # Apple and Microsoft stocks
    start_date = '2020-01-01'
    end_date = '2021-01-01'
    window_size = 30  # Number of lagged days to use as features
    
    for ticker in tickers:
        # Download stock data
        stock_data = download_stock_data(ticker, start_date, end_date)
        
        # Preprocess data
        stock_data = preprocess_data(stock_data)
        
        # Create features and target variable
        X, y = create_features_target(stock_data, target_column='Close', window_size=window_size)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = train_model(X_train, y_train)
        
        # Evaluate the model
        mse = evaluate_model(model, X_test, y_test)
        print(f"Mean Squared Error for {ticker}: {mse}")

# Run the program
if __name__ == "__main__":
    main()
