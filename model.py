

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="max")
        if data.empty:
            print(f"No data available for the symbol: {symbol}")
            return None
        else:
            # print(f"Historical Data for {symbol}:\n{data.head()}\n")
            print(f"Historical Data for {symbol}: {data}\n")
            return data
    except Exception as e:
        print(f"Error occurred while fetching data for {symbol}: {e}")
        return None

def create_features(data):
    if data is None or data.empty:
        return None, None

    data = data.copy()  # Create a copy of the data DataFrame

    # Calculate moving averages

    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["MA200"] = data["Close"].rolling(window=200).mean()

   #Create target variable ("Tomorrow price and binary target)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

    data = data.dropna()  # Drop NaN values from the entire DataFrame

    #Prepare features X and target y
    features = data.drop(["Close", "Tomorrow"], axis=1)
    target = data["Tomorrow"]

    return features, target


def train_model(features, target):
    imputer = SimpleImputer(strategy="mean")
    imputed_features = imputer.fit_transform(features)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(imputed_features, target)
    return model, imputer

def predict(model, imputer, new_data):
    imputed_data = imputer.transform(new_data)
    predictions = model.predict(imputed_data)
    return predictions

# def backtest(data, model, imputer):
#     if data is None or data.empty:
#         print("No data available for backtesting.")
#         return None

def backtest(data, model, imputer):
    tscv = TimeSeriesSplit(n_splits=3)
    all_predictions = []

    # Create features and target using all available data
    features, target = create_features(data)

    if features is None or target is None or features.empty or target.empty:
        print("Could not create valid features and target.")
        return None

    imputed_features = imputer.fit_transform(features)  # Fit the imputer on all features

    for train_index, test_index in tscv.split(features):
        train_features = imputed_features[train_index]
        train_target = target.iloc[train_index]
        test_features = imputed_features[test_index]
        test_target = target.iloc[test_index]

        # Train the model using the current training split
        model.fit(train_features, train_target)

        # Make predictions using the current test split
        predictions = model.predict(test_features)

        # Check if lengths of test_target and predictions match
        if len(test_target) != len(predictions):
            print(f"Length mismatch in predictions for split {train_index} and {test_index}. Skipping this split.")
            continue

        # Construct DataFrame for backtest results
        predictions_df = pd.DataFrame({
            "Date": data.index[test_index],
            "Actual": test_target.values,
            "Predicted": predictions
        }).set_index("Date")

        all_predictions.append(predictions_df)

    if all_predictions:
        backtest_results = pd.concat(all_predictions)
        return backtest_results
    else:
        print("No valid backtest results generated.")
        return None



def evaluate_model(backtest_results):
    if "Actual" in backtest_results.columns and "Predicted" in backtest_results.columns:
        mse = mean_squared_error(backtest_results["Actual"], backtest_results["Predicted"])
        mae = mean_absolute_error(backtest_results["Actual"], backtest_results["Predicted"])
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
    else:
        print("Actual or Predicted column not found in backtest_results. Skipping evaluation.")

def visualize_predictions(data, backtest_results):
    if data is None or backtest_results is None:
        print("No data or backtest results available for visualization.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Actual")
    plt.plot(backtest_results.index, backtest_results["Predicted"], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Actual vs. Predicted Stock Prices")
    plt.legend()
    plt.show()