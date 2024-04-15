

from model import get_stock_data, create_features, train_model, predict, backtest, evaluate_model, visualize_predictions
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def main():

    while True:
        symbol = input("Enter a stock symbol (or 'exit' to quit): ")
        if symbol.lower() == "exit":
            break

        #Retrieve stock data
        data = get_stock_data(symbol)

        if data is None or data.empty:
            print(f"No data available for the symbol: {symbol}")
            continue

        #Create feature and target
        features, target = create_features(data)

        if features is None or target is None or features.empty or target.empty:
            print(f"Could not create valid features for {symbol}")
            continue

        #Train the model
        model, imputer = train_model(features, target)

        if model is None:
            print("Failed to train the model.")
            continue

        #Make predictions and perform backtesting
        # predictions = predict(model, imputer, features)
        backtest_results = backtest(data, model, imputer)

        if backtest_results is not None:

            print("Backtest results:")
            print(backtest_results.head())

            #Evaluate model performance
            evaluate_model(backtest_results)

            #Visualize predictions
            visualize_predictions(data, backtest_results)  # Pass backtest_results instead of predictions
        else:
            print("Error occurred during backtesting. Skipping evaluation and visualization.")

        # Print tomorrow's predicted price
        if backtest_results is not None and not backtest_results.empty:
            last_date = data.index[-1]
            tomorrow_date = last_date + pd.DateOffset(days=1)
            tomorrow_price = backtest_results.loc[tomorrow_date, "Predicted"]
            print(f"Predicted price for {symbol} on {tomorrow_date.date()}: {tomorrow_price:.2f}")
        else:
            print("Could not retrieve tomorrow's predicted price.Error occurred during")

    # Debug: Inspect index of test_target and predictions
    print("test_target index:", test_target.index)
    print("predictions index:", backtest_results.index)

    print("NaNs in test_target:", test_target.isnull().sum())
    print("NaNs in predictions:", backtest_results["Predicted"].isnull().sum())



if __name__ == "__main__":

    main()