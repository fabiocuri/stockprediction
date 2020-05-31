import numpy as np
import datetime as dt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from firebase_actions import *

def data_stock(stock_data, features):
    ''' Filters selected features '''

    features = features.split()
    df = stock_data
    columns = ['Close']
    
    if len(features) > 0:
        for i in range(len(features)):
            columns.append(features[i])

    df = df[columns]
    df = df.loc[:,~df.columns.duplicated()]

    return df
    
def get_sequences(matrix, steps):
    ''' Generates sequences for LSTM '''
    
    x_data, y = [], []
    
    for i in range(matrix.shape[0]-steps):
        x = []
        for j in range(steps):
            x.append(matrix[i+j,:])
        x_data.append(x)
        y.append(matrix[i+j+1,0])
            
    return np.asarray(x_data), np.asarray(y)
    
def get_last_sequence(matrix, steps):
    ''' Generates last sequence for prediction '''

    x = []
    for i in list((np.array(range(steps)) + 1) * -1)[::-1]:
        x.append(matrix[i,:])
        
    return np.asarray(x)

def split_data(new_df, steps, training, tune_boolean):
    ''' Splits data for training and test '''
    
    data_df = new_df.as_matrix().astype(float)
    rows = round(data_df.shape[0] * training)
    
    if tune_boolean:
        train_data = data_df[:rows,:]
    else:
        train_data = data_df
        
    # Train the Scaler with training data and smooth data
    scaler = MinMaxScaler()
    smoothing_window_size = int(np.floor(0.25*train_data.shape[0]))
    
    for di in range(0, int(np.floor(0.5*train_data.shape[0])), smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
    
    if tune_boolean:
        test_data = data_df[rows:,:]
        test_data = scaler.transform(test_data)
    else:
        test_data = train_data
        
    data = {}
    data['train_x'], data['train_y'] = get_sequences(train_data, steps)
    data['test_x'], data['test_y'] = get_sequences(test_data, steps)
    last_sequence = get_last_sequence(test_data, steps)
    last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])

    return data, scaler, last_sequence
    
def feature_selection(data, columns):
    ''' Performs feature selection '''
    
    data['train_x'] = data['train_x'][:,-1,:]
    mi = mutual_info_regression(data['train_x'], data['train_y'])
    mi /= np.max(mi)
    selected_features=[]
    
    for key, name in zip(mi, columns):
        if key > 0.2:
            selected_features.append(name)
        
    return selected_features

def run(data, lstm_size, batch_size, learning_rate, scaler, last_sequence, tune_boolean):
    ''' Run an instance of the model '''
        
    if tune_boolean:
        X, Y = data['train_x'], data['train_y']
        VAL_X, VAL_Y = data['test_x'], data['test_y']
    else:
        X = np.concatenate((data['train_x'], data['test_x']), axis=0)
        Y = np.concatenate((data['train_y'], data['test_y']), axis=0)
        VAL_X, VAL_Y = X, Y
        
    # Model
    model = Sequential()
    model.add(LSTM(int(lstm_size), input_shape=(data['train_x'].shape[1], data['train_x'].shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer="Adam")
    model.fit(X, Y, epochs=500, validation_data=(VAL_X, VAL_Y), batch_size=int(batch_size), verbose=0, callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)])

    # Predict next day price
    price = model.predict(last_sequence)
    price_ = np.append(price, [-0.5])
    price = scaler.inverse_transform([price_])[0][0]

    return price
    
def evaluate_backtesting(output):
    ''' Evaluates backtesting results '''

    score = 0
    
    for i in output:
        if output[i]:
            score += 1

    return score
    
def get_dates(stock_data):
    ''' Gets today and next labor dates'''

    value_today = list(stock_data['Close'])[-1]
    last_date = stock_data.index[-1]
    last_date = dt.datetime.strptime(str(last_date), '%Y-%m-%d %H:%M:%S')
    next_date = (last_date + dt.timedelta(days=1))
    
    while next_date.weekday() == 5 or next_date.weekday() == 6:
        next_date = next_date + dt.timedelta(days=1)
                        
    next_date = next_date.strftime('%Y-%m-%d')

    return value_today, next_date

def hyperparameter_tuning(stock, stock_data, length_backtesting, features, steps, training, period, years, num_exps, db, index):
    ''' Hyper-parameter tuning with back-testing'''
    
    tune_boolean = True
    best_score = -1
    
    for lstm_size in [10]:
        for batch_size in [128]:
            for learning_rate in [0.01]:
            
                # Back-testing
                output = {}
                
                for i in list((np.array(range(length_backtesting)) + 1) * -1):
                    value_next_day = list(stock_data['Close'])[i]
                    stock_data_ = stock_data[:i]
                    value_today, next_date = get_dates(stock_data_)
                    finalprice_mean, finaltrend_mean = 0, 0
                    
                    new_df = data_stock(stock_data_, features)
                    data, scaler, last_sequence = split_data(new_df, steps, training, tune_boolean)
                    selected_features = feature_selection(data, new_df.columns)
                    new_df = new_df[selected_features]
                    data, scaler, last_sequence = split_data(new_df, steps, training, tune_boolean)

                    for i in range(num_exps):
                        price = run(data, lstm_size, batch_size, learning_rate, scaler, last_sequence, tune_boolean)
                        finalprice_mean+=price
                        finaltrend_mean+=((price-value_today)/value_today)*100
                        
                    finalprice_mean/=num_exps
                    finaltrend_mean/=num_exps

                    # Evaluate back-testing
                    real_trend = ((value_next_day-value_today)/value_today)*100
                    sign_predicted = finaltrend_mean > 0
                    sign_true = real_trend > 0
                    is_match = sign_predicted == sign_true
                    output[next_date] = is_match

                score = evaluate_backtesting(output)
                
                if score > best_score:
                    best_params = (str(lstm_size), str(batch_size), str(learning_rate), str(selected_features))
                    best_score = score
  
    accuracy = str(round(best_score * 100 / length_backtesting, 2))
    data = {"LSTM size": best_params[0], "Batch size": best_params[1], "Learning rate": best_params[2], "Selected features": best_params[3], "Back-testing accuracy": accuracy}
    for index_ in index:
        export_firebase(data, stock, db, '{}_Params'.format(index_))
    print('STEP [2/3]: Model tuned and exported to Firebase!')
    
def predict_tomorrow(stock, stock_data, num_exps, steps, training, db, params, index):
    ''' Predict price for the next day '''
    
    tune_boolean = False
    
    if not params:
        lstm_size, batch_size, learning_rate, features = 10, 128, 0.01, 'Close S&P500'
    else:
        lstm_size, batch_size, learning_rate, features = params['lstm_size'], params['batch_size'], params['learning_rate'], params['selected_features'].split("'")[1]
        
    new_df = data_stock(stock_data, features)
    data, scaler, last_sequence = split_data(new_df, steps, training, tune_boolean)
    selected_features = new_df.columns
    
    value_today, next_date = get_dates(stock_data)
    finalprice_mean, finaltrend_mean = 0, 0
    
    for i in range(num_exps):
        price = run(data, lstm_size, batch_size, learning_rate, scaler, last_sequence, tune_boolean)
        finalprice_mean+=price
        finaltrend_mean+=((price-value_today)/value_today)*100
        
    finalprice_mean/=num_exps
    finaltrend_mean/=num_exps
    data = {"Date": next_date, "Price": round(finalprice_mean,2), "Trend": round(finaltrend_mean,2)}
    for index_ in index:
        export_firebase(data, stock, db, '{}_Predictions'.format(index_))
    print('STEP [3/3]: Predictions exported to Firebase!')
    
if '__main__' == __name__:
    print('')
