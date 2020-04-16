# coding: utf-8

import os
import random
import time
import argparse
import pandas as pd
import datetime as dt
import numpy as np
import urllib.request, json
from collections import deque
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from yahoo_fin import stock_info as si
from IPython.display import clear_output
from sklearn import preprocessing
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, BatchNormalization, Dropout

def load_stock_data(api_key, ticker):
    """
    Loads stock data from either Alpha Vantage or Yahoo Finance platform.
    Params:
        api_key (str): key to use the Alpha Vantage API. (https://www.alphavantage.co/support/#api-key)
        ticker (str): the ticker you want to load, examples include AAPL, TESL, etc
    """

    # JSON file with all the stock market data for a given ticker within the last 20 years
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

    # Save data to this file
    file_to_save = './data/stock_market_data-%s.csv'%ticker

    try:
        # Alpha Vantage
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # extract stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['1. open']),float(v['2. high']),
                            float(v['3. low']),float(v['4. close']),float(v['5. volume'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        print('Data saved to : %s'%file_to_save)  
        df = df.sort_values(by=['Date'])
        df.to_csv(file_to_save)
    except:
        try:
            # Yahoo Finance platform
            df = si.get_data(ticker)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns=['Open', 'High', 'Low', 'Close', 'Volume']
            df.insert(0, 'Date', df.index)
            df.index = list(range(len(df)))
            print('Data saved to : %s'%file_to_save)        
            df.to_csv(file_to_save)
        except:
            print('No data was found. Please try a valid ticker name.')
            df = ''
    
    return df

def preprocess_data(df, n_steps=50, scale=True, shuffle=True, lookup_step=1, 
                test_size=0.2, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume']):
    """
    Performs scaling, shuffling, normalizing and splitting of stock dataframe.
    Params:
        df (pd.DataFrame): dataframe with stock prices for a given ticker
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model
    """

    result = {}
    result['df'] = df.copy()

    for col in feature_columns:
        assert col in df.columns

    if scale:
        column_scaler = {}
        # Normalize the data
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        result["column_scaler"] = column_scaler

    # Add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Close'].shift(-lookup_step)

    # Get last sequence
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    result['last_sequence'] = last_sequence
    
    # Build X and y
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
   
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)

    return result

def create_model(input_length, units=256, n_layers=2, learning_rate=0.001):
    """
        Creates a LSTM model.
        Params:
        input_length (int): its value is equal to n_steps
        units (int): LSTM size
        n_layers (int): number of layers in the network
        dropout (float): regularization factor in %
        loss (str): measure for gradient descent
        learning_rate (float): the learning rate for gradient descent
    """
    
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # First layer
            model.add(LSTM(units, return_sequences=True, input_shape=(None, input_length)))
        elif i == n_layers - 1:
            # Last layer
            model.add(LSTM(units, return_sequences=False))
        else:
            # Hidden layers
            model.add(LSTM(units, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
    
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", metrics=["mean_absolute_error"], optimizer=Adam(learning_rate=learning_rate))

    return model

def run_model(model, data):
    
    earlystopping = EarlyStopping(monitor='val_mean_absolute_error', patience=10, verbose=0, mode='min')

    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[earlystopping],
                        verbose=1)
    
    MAE_score = model.evaluate(data["X_test"], data["y_test"])[1]
    
    return model, MAE_score

def plot_graph(model, data):
    
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["Close"].inverse_transform(y_pred))
    plt.plot(y_test[-200:], c='b')
    plt.plot(y_pred[-200:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

def predict(model, data, classification=False):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][:N_STEPS]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["Close"].inverse_transform(prediction)[0][0]
    return predicted_price

if '__main__' == __name__:

    parser = argparse.ArgumentParser(description='Stock details')
    parser.add_argument('--ticker', type=str, help='The ticker you want to load, examples include AAPL, TESL, etc')
    parser.add_argument('--api_alpha_vantage', type=str, help='Key to use the Alpha Vantage API. (https://www.alphavantage.co/support/#api-key)')
    args = parser.parse_args()
    
    # Load stock data
    TICKER=args.ticker
    API_ALPHA_VANTAGE=args.api_alpha_vantage
    df = load_stock_data(api_key=API_ALPHA_VANTAGE, ticker=TICKER)

    # Create folders
    if not os.path.isdir("./results"):
        os.mkdir("./results")
    if not os.path.isdir("./data"):
        os.mkdir("./data")

    # Preprocess data
    N_STEPS = 50
    LOOKUP_STEP = 30
    TEST_SIZE = 0.2
    FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    DATA_TRAIN = preprocess_data(df, n_steps=N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                                 feature_columns=FEATURE_COLUMNS)
    DATA_EVALUATE = preprocess_data(df, n_steps=N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                                    feature_columns=FEATURE_COLUMNS, shuffle=False)

    # Build, train a LSTM and do hyper-parameter tuning
    MAE_best = 10000
    EPOCHS = 500
    lookback=[]

    for i in range(30):

        date_now = time.strftime("%Y-%m-%d")

        N_LAYERS = random.choice([2, 3, 4])
        UNITS = random.choice([128, 256])
        LEARNING_RATE = np.power(10,(-4 * np.random.rand()))
        BATCH_SIZE = random.choice([32, 64, 128])

        model_name = '{}-{}-seq-{}-step-{}-layers-{}-units-{}-lr-{}-batch-{}'.format(TICKER,date_now,N_STEPS,LOOKUP_STEP,N_LAYERS,UNITS,LEARNING_RATE,BATCH_SIZE)
        
        if model_name not in lookback:
            lookback.append(model_name)

            # Create the model
            MODEL = create_model(N_STEPS, units=UNITS, n_layers=N_LAYERS, learning_rate=LEARNING_RATE)

            # Run
            model, MAE_score = run_model(model=MODEL, data=DATA_TRAIN)

            if MAE_score < MAE_best:
                MAE_best = MAE_score
                best_model = model
                #clear_output(wait=True)
                print('Best MAE: ' + str(MAE_best))

    best_model.save(os.path.join("./results/LSTM_best_model.h5"))

    # Predict future price
    #future_price = predict(best_model, DATA_EVALUATE)
    #print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}")

    # Plot test predictions
    plot_graph(best_model, DATA_EVALUATE)
