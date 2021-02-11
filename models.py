import numpy as np
import pandas as pd
import datetime as dt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from get_data import *
from firebase_actions import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def format_floats(value, n):
    return str(round(value, n))

def impute_missing_values(stock_data):
    """ 
    Missing values imputation.
    """

    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputted = imputer.fit_transform(stock_data.values)

    imputted_df = pd.DataFrame(imputted)
    assert stock_data.shape == imputted_df.shape
    assert imputted_df.isnull().values.any() == 0

    imputted_df.columns = stock_data.columns
    imputted_df.index = stock_data.index
    stock_data = imputted_df

    return stock_data


def get_dates(stock_data):
    """
    Gets today value and next labor date.
    """

    last_date = stock_data.index[-1]
    last_date = dt.datetime.strptime(str(last_date), '%Y-%m-%d %H:%M:%S')
    next_date = (last_date + dt.timedelta(days=1))

    while next_date.weekday() == 5 or next_date.weekday() == 6:
        next_date = next_date + dt.timedelta(days=1)

    next_date = next_date.strftime('%Y-%m-%d')

    return next_date


def get_sequences(data, steps):
    """
    Generates sequences for LSTM.
    """

    x_data, y = [], []

    for i in range(data.shape[0]-steps):
        x = []
        for j in range(steps):
            x.append(data[i+j, :])
        x_data.append(x)
        y.append(data[i+j+1, -1])

    return np.asarray(x_data), np.asarray(y)


def get_last_sequence(data, steps):
    """
    Generates last sequence for prediction.
    """

    last_sequence = []
    for i in list((np.array(range(steps)) + 1) * -1)[::-1]:
        last_sequence.append(data[i, :])

    last_sequence = np.asarray(last_sequence)
    last_sequence = last_sequence.reshape(
        1, last_sequence.shape[0], last_sequence.shape[1])

    return last_sequence


def split_data(new_df, training, tune_boolean, steps):
    """
    Splits data for training and test.
    """

    df_copy = new_df.copy()

    gain_loss = df_copy['GAIN_LOSS']
    df_copy.drop(columns=['GAIN_LOSS'], inplace=True)

    data_df = df_copy.values.astype(float)
    rows = round(data_df.shape[0] * training)

    if tune_boolean:
        train_data = data_df[:rows, :]
        train_gain_loss = np.asarray(list(gain_loss[:rows]))
    else:
        train_data = data_df
        train_gain_loss = np.asarray(list(gain_loss))

    # Train the Scaler with training data and smooth data
    scaler = MinMaxScaler()
    smoothing_window_size = int(np.floor(0.25*train_data.shape[0]))

    for di in range(0, int(np.floor(0.5*train_data.shape[0])), smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size, :])
        train_data[di:di+smoothing_window_size,
                   :] = scaler.transform(train_data[di:di+smoothing_window_size, :])

    scaler.fit(train_data[di+smoothing_window_size:, :])
    train_data[di+smoothing_window_size:,
               :] = scaler.transform(train_data[di+smoothing_window_size:, :])

    if tune_boolean:
        test_data = data_df[rows:, :]
        test_data = scaler.transform(test_data)
        test_gain_loss = np.asarray(gain_loss[rows:])
    else:
        test_data = train_data
        test_gain_loss = np.asarray(gain_loss)

    train_gain_loss.resize(train_gain_loss.shape[0], 1)
    test_gain_loss.resize(test_gain_loss.shape[0], 1)

    train_data = np.concatenate((train_data, train_gain_loss), axis=1)
    test_data = np.concatenate((test_data, test_gain_loss), axis=1)

    data = {}
    data['train_x'], data['train_y'] = get_sequences(train_data, steps)
    data['test_x'], data['test_y'] = get_sequences(test_data, steps)

    last_sequence = get_last_sequence(test_data, steps)

    return data, last_sequence


def feature_selection(data, columns):
    """ 
    Performs feature selection.
    """

    data['train_x'] = data['train_x'][:, -1, :]
    mi = mutual_info_regression(data['train_x'], data['train_y'])
    mi /= np.max(mi)
    selected_features = []

    for key, name in zip(mi, columns):
        if key > 0.2:
            if name != 'Close' and name != 'GAIN_LOSS':
                selected_features.append(name)

    selected_features.append('Close')
    selected_features.append('GAIN_LOSS')

    return selected_features


def run(data, last_sequence, tune_boolean, lstm_size, batch_size, learning_rate):
    """ 
    Run an instance of the model
    """

    if tune_boolean:
        X, Y = data['train_x'], data['train_y']
        VAL_X, VAL_Y = data['test_x'], data['test_y']
    else:
        X = np.concatenate((data['train_x'], data['test_x']), axis=0)
        Y = np.concatenate((data['train_y'], data['test_y']), axis=0)
        VAL_X, VAL_Y = X, Y

    # Model
    model = Sequential()
    model.add(LSTM(int(lstm_size), input_shape=(
        data['train_x'].shape[1], data['train_x'].shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer="Adam")
    model.fit(X, Y, epochs=500, validation_data=(VAL_X, VAL_Y), batch_size=int(batch_size),
              verbose=0, callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)])

    # Predict next day gain_loss
    gain_loss = model.predict(last_sequence)

    return gain_loss[0][0]


def evaluate_backtesting(output):
    """ 
    Evaluates backtesting results 
    """

    score = 0

    for date_entry in output:

        if output[date_entry]:

            score += 1

    return score


def hyperparameter_tuning(stock, stock_data, years, length_backtesting, steps, training, db, index):
    """
    Hyper-parameter tuning with back-testing.
    """

    tune_boolean = True
    best_score = -1

    for lstm_size in [10]:

        for batch_size in [128]:

            for learning_rate in [0.01]:

                # Back-testing
                output = {}

                for i in list((np.array(range(length_backtesting)) + 1) * -1):

                    value_next_day = list(stock_data['GAIN_LOSS'])[i]

                    stock_data_ = stock_data[:i]

                    next_date = get_dates(stock_data_)

                    data, last_sequence = split_data(
                        stock_data_, training, tune_boolean, steps)

                    if i == -1:

                        selected_features = feature_selection(
                            data, stock_data_.columns)
                        model_features = selected_features

                    stock_data_ = stock_data_[model_features]
                    assert list(stock_data_.columns)[-1] == 'GAIN_LOSS'

                    data, last_sequence = split_data(
                        stock_data_, training, tune_boolean, steps)

                    gain_loss = run(
                        data, last_sequence, tune_boolean, lstm_size, batch_size, learning_rate)

                    # Evaluate back-testing
                    sign_predicted = gain_loss > 0
                    sign_true = value_next_day > 0
                    is_match = sign_predicted == sign_true
                    output[next_date] = is_match

                score = evaluate_backtesting(output)

                if score > best_score:

                    best_params = (str(lstm_size), str(batch_size), str(
                        learning_rate), str(model_features))
                    best_score = score

    accuracy = str(round(best_score * 100 / length_backtesting, 2))
    hyperparams = {"LSTM size": best_params[0], "Batch size": best_params[1],
                   "Learning rate": best_params[2], "Selected features": best_params[3], "Back-testing accuracy": accuracy}

    for index_ in index:

        export_firebase(data=hyperparams, stock=stock, db=db,
                        folder='{}_Params'.format(index_))

    print('STEP [2/3]: Model tuned and exported to Firebase!')


def predict_tomorrow(stock, stock_data, steps, training, db, index, params):
    """ 
    Predict gain_loss and price for the next day.
    """
    
    columns = list(stock_data.columns)
    
    last_stats = stock_data.iloc[-1]
    
    data_last_stats = {}
    
    for entry in columns:
    
        data_last_stats['LAST_' + entry] = format_floats(last_stats[entry], 4)

    tune_boolean = False

    last_price = float(stock_data['Close'][-1])

    lstm_size, batch_size, learning_rate, model_features = params['lstm_size'], params['batch_size'], params['learning_rate'], params['selected_features'].replace(
        "'", '').replace("[", '').replace("]", '').split(', ')

    next_date = get_dates(stock_data)

    stock_data = stock_data[model_features]
    assert list(stock_data.columns)[-1] == 'GAIN_LOSS'
    
    data, last_sequence = split_data(
        stock_data, training, tune_boolean, steps)

    gain_loss = run(data, last_sequence, tune_boolean,
                    lstm_size, batch_size, learning_rate)

    next_price = (1 + gain_loss) * last_price
    
    last_row = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
    stock_data = stock_data.append(last_row)
    stock_data['GAIN_LOSS'][-1] = gain_loss
    
    stock_data['Close'][-1] = next_price
    
    if 'Adj Close' not in list(stock_data.columns):
    
        stock_data['Adj Close'] = list(stock_data['Close'])
        
    stock_data['Adj Close'][-1] = next_price
    
    if 'Open' not in list(stock_data.columns):
    
        stock_data['Open'] = list(range(len(stock_data)))
        stock_data['Open'][1:] = stock_data['Close'][:-1]
        
    stock_data['Open'][-1] = last_price
    
    if 'Low' not in list(stock_data.columns):
    
        stock_data['Low'] = stock_data['Open']
        
    stock_data['Low'][-1] = last_price
        
    if 'High' not in list(stock_data.columns):
    
        stock_data['High'] = stock_data['Close']
        
    stock_data['High'][-1] = next_price
    
    assert 'Open' in stock_data.columns
    assert 'High' in stock_data.columns
    assert 'Low' in stock_data.columns
    assert 'Close' in stock_data.columns
    
    stock_data = add_sma(df=stock_data, period=20)
    stock_data = add_ema(df=stock_data, period=14)
    stock_data = add_k(df=stock_data, period=5)
    stock_data = add_macd(df=stock_data, period_fast=12, period_slow=26, period_signal=9)
    stock_data = add_bb(df=stock_data, period=14)
    stock_data = add_rsi(df=stock_data, period=14)
    stock_data = add_fibonacci(df=stock_data, lookback_up=235, lookback_down=135)
    stock_data = add_ichimoku(df=stock_data)
    stock_data = add_std(df=stock_data, period=14)
    stock_data = add_adx(df=stock_data)
    stock_data = add_r(df=stock_data, period=14)
    
    stock_data_ = stock_data[['Open', 'High', 'Low', 'Close']]
    
    stock_data_ = add_inverted_hammer(df=stock_data_)
    stock_data_ = add_hammer(df=stock_data_)
    stock_data_ = add_hanging_man(df=stock_data_)
    stock_data_ = add_bearish_harami(df=stock_data_)
    stock_data_ = add_bullish_harami(df=stock_data_)
    stock_data_ = add_dark_cloud_cover(df=stock_data_)
    stock_data_ = add_doji(df=stock_data_)
    stock_data_ = add_doji_star(df=stock_data_)
    stock_data_ = add_dragonfly_doji(df=stock_data_)
    stock_data_ = add_gravestone_doji(df=stock_data_)
    stock_data_ = add_bearish_engulfing(df=stock_data_)
    stock_data_ = add_bullish_engulfing(df=stock_data_)
    stock_data_ = add_morning_star(df=stock_data_)
    stock_data_ = add_morning_star_doji(df=stock_data_)
    stock_data_ = add_piercing_pattern(df=stock_data_)
    stock_data_ = add_rain_drop(df=stock_data_)
    stock_data_ = add_rain_drop_doji(df=stock_data_)
    stock_data_ = add_star(df=stock_data_)
    stock_data_ = add_shooting_star(df=stock_data_)
    stock_data_ = add_evening_star(df=stock_data_)
    
    predicted_stats = stock_data.iloc[-1]
    predicted_candlesticks = stock_data_.iloc[-1]
    
    rsi_trend = (round(predicted_stats['RSI'], 4) - float(data_last_stats['LAST_RSI']))/float(data_last_stats['LAST_RSI'])*100
    macd_trend = (round(predicted_stats['MACD'], 4) - float(data_last_stats['LAST_MACD']))/float(data_last_stats['LAST_MACD'])*100

    predictions = {"Date": next_date,
                   "PRED_Price": format_floats(next_price, 2),
                   "PRED_GAIN_LOSS": format_floats(gain_loss, 4),
                   "PRED_SMA": format_floats(predicted_stats['SMA'], 4),
                   "PRED_EMA": format_floats(predicted_stats['EMA'], 4),
                   "PRED_%K": format_floats(predicted_stats['%K'], 4),
                   "PRED_MACD": format_floats(predicted_stats['MACD'], 4),
                   "PRED_BB-Upper": format_floats(predicted_stats['BB-Upper'], 4),
                   "PRED_BB-Lower": format_floats(predicted_stats['BB-Lower'], 4),
                   "PRED_RSI": format_floats(predicted_stats['RSI'], 4),
                   "PRED_Fibonacci_min": format_floats(predicted_stats['Fibonacci_min'], 4),
                   "PRED_Fibonacci_max": format_floats(predicted_stats['Fibonacci_max'], 4),
                   "PRED_tenkan_sen": format_floats(predicted_stats['tenkan_sen'], 4),
                   "PRED_kijun_sen": format_floats(predicted_stats['kijun_sen'], 4),
                   "PRED_senkou_span_a": format_floats(predicted_stats['senkou_span_a'], 4),
                   "PRED_senkou_span_b": format_floats(predicted_stats['senkou_span_b'], 4),
                   "PRED_chikou_span": format_floats(predicted_stats['chikou_span'], 4),
                   "PRED_STD_CLOSE": format_floats(predicted_stats['STD_CLOSE'], 4),
                   "PRED_ADX": format_floats(predicted_stats['ADX'], 4),
                   "PRED_%R": format_floats(predicted_stats['%R'], 4),
                   "PRED_Inverted Hammer": str(predicted_candlesticks['Inverted Hammer']),
                   "PRED_Hammer": str(predicted_candlesticks['Hammer']),
                   "PRED_Hanging Man": str(predicted_candlesticks['Hanging Man']),
                   "PRED_Bearish Harami": str(predicted_candlesticks['Bearish Harami']),
                   "PRED_Bullish Harami": str(predicted_candlesticks['Bullish Harami']),
                   "PRED_Dark Cloud Cover": str(predicted_candlesticks['Dark Cloud Cover']),     
                   "PRED_DOJI": str(predicted_candlesticks['DOJI']),
                   "PRED_DOJI Star": str(predicted_candlesticks['DOJI Star']),
                   "PRED_Dragonfly DOJI": str(predicted_candlesticks['Dragonfly DOJI']),
                   "PRED_Gravestone DOJI": str(predicted_candlesticks['Gravestone DOJI']),
                   "PRED_Bearish Engulfing": str(predicted_candlesticks['Bearish Engulfing']),
                   "PRED_Bullish Engulfing": str(predicted_candlesticks['Bullish Engulfing']),
                   "PRED_Morning Star": str(predicted_candlesticks['Morning Star']),
                   "PRED_Morning Star DOJI": str(predicted_candlesticks['Morning Star DOJI']),
                   "PRED_Piercing": str(predicted_candlesticks['Piercing']),
                   "PRED_Rain Drop": str(predicted_candlesticks['Rain Drop']),
                   "PRED_Rain Drop DOJI": str(predicted_candlesticks['Rain Drop DOJI']),
                   "PRED_Star": str(predicted_candlesticks['Star']),    
                   "PRED_Shooting Star": str(predicted_candlesticks['Shooting Star']),
                   "PRED_Evening Star": str(predicted_candlesticks['Evening Star']),
                   "PRED_RSI_TREND": format_floats(rsi_trend, 4),
                   "PRED_MACD_TREND": format_floats(macd_trend, 4)}
    
    output = {**predictions, **data_last_stats}

    for index_ in index:

        export_firebase(data=output, stock=stock, db=db,
                        folder='{}_Predictions'.format(index_))

    print('STEP [3/3]: Predictions exported to Firebase!')


if '__main__' == __name__:
    print('')
