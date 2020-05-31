import json
import urllib
import pandas as pd
import datetime as dt
from pandas_datareader import data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_alpha_vantage(stock, alphavantage_key):
    ''' Get data from Alpha Vantage '''

    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s" % (stock, alphavantage_key)

    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        # extract stock market data
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        for k, v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(), float(v['1. open']), float(v['2. high']),
                        float(v['3. low']), float(v['4. close']), float(v['5. volume'])]
            df.loc[-1, :] = data_row
            df.index = df.index + 1
    df = df.sort_values(by=['Date'])

    return df

def get_yahoo(stock, years):
    ''' Get data from Yahoo Finance '''

    start = (dt.datetime.today() - dt.timedelta(days=365 * years)).strftime('%Y-%m-%d')
    end = dt.datetime.today().strftime('%Y-%m-%d')
    df = data.DataReader(stock, 'yahoo', start, end)

    return df, start, end

def add_k(df, period):
    ''' Calculate Stochastic Oscillator (%K) '''

    df['L14'] = df.Low.rolling(period).min()
    df['H14'] = df.High.rolling(period).max()
    df['%K'] = ((df.Close - df.L14) / (df.H14 - df.L14)) * 100
    df.drop(columns=['L14', 'H14'], inplace=True)

    return df

def add_r(df, period):
    ''' Calculate Larry William indicator (%R) '''

    df['HH'] = df.High.rolling(period).max()
    df['LL'] = df.Low.rolling(period).min()
    df['%R'] = ((df.HH - df.Close) / (df.HH - df.LL)) * (-100)
    df.drop(columns=['HH', 'LL'], inplace=True)

    return df

def add_rsi(df, period):
    ''' Calculate RSI values '''

    df['Change'] = df.Close - df.Open
    df['Gain'] = df.Change[df.Change > 0]
    df['Loss'] = df.Change[df.Change < 0] * (-1)
    df.drop(columns=['Change'], inplace=True)
    df.Gain.fillna(0, inplace=True)
    df.Loss.fillna(0, inplace=True)
    df['Again'] = df.Gain.rolling(period).mean()
    df['Aloss'] = df.Loss.rolling(period).mean()
    df['RS'] = df.Again / df.Aloss
    df['RSI'] = 100 - (100 / (1 + (df.Again / df.Aloss)))
    df.drop(columns=['Gain', 'Loss', 'Again', 'Aloss', 'RS'], inplace=True)

    return df

def add_vix(df, alphavantage_key):
    ''' Calculate VIX index '''

    df_feature = get_alpha_vantage('VIX', alphavantage_key)
    df_feature.index = df_feature['Date']
    df_feature = df_feature[['Close']]
    df_feature.columns = ['VIX']
    df = df.join(df_feature)

    return df

def add_sp500(df, years):
    ''' Calculate S&P500 index '''

    start = (dt.datetime.today() - dt.timedelta(days=365 * years)).strftime('%Y-%m-%d')
    end = dt.datetime.today().strftime('%Y-%m-%d')
    df_feature = data.DataReader(['sp500'], 'fred', start, end)
    df_feature.columns = ['S&P500']
    df = df.join(df_feature)

    return df

def get_historical_data(stock, years, period, features, alphavantage_key):
    ''' Retrieves historical data with financial features '''

    df, start, end = get_yahoo(stock, years)

    if '%K' in features:
        df = add_k(df, period)

    if '%R' in features:
        df = add_r(df, period)

    if 'RSI' in features:
        df = add_rsi(df, period)

    if 'VIX' in features:
        df = add_vix(df, alphavantage_key)

    if 'S&P500' in features:
        df = add_sp500(df, years)

    df = df.dropna()
    print('STEP [1/3]: Data retrieved!')

    return df

if '__main__' == __name__:
    print('')
