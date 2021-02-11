import json
import urllib
import pandas as pd
import datetime as dt
from pandas_datareader import data
from stockstats import StockDataFrame
import numpy as np
import warnings
import candlestick
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_alpha_vantage(stock, alphavantage_key):
    """
    Get data from Alpha Vantage.
    """

    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s" % (
        stock, alphavantage_key)

    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        # extract stock market data
        data = data['Time Series (Daily)']
        df = pd.DataFrame(
            columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        for k, v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(), float(v['1. open']), float(v['2. high']),
                        float(v['3. low']), float(v['4. close']), float(v['5. volume'])]
            df.loc[-1, :] = data_row
            df.index = df.index + 1
    df = df.sort_values(by=['Date'])

    return df


def get_yahoo(stock, years):
    """
    Get data from Yahoo Finance.
    """

    start = (dt.datetime.today() -
             dt.timedelta(days=365 * years)).strftime('%Y-%m-%d')
    end = dt.datetime.today().strftime('%Y-%m-%d')
    df = data.DataReader(stock, 'yahoo', start, end)

    return df, start, end


def add_sma(df, period):
    """
    Simple Moving Average of closing prices over a period.
    """

    df['SMA'] = df['Close'].rolling(window=period).mean()

    return df


def add_ema(df, period):
    """
    Exponential moving average over period days.
    """
    
    EMA = []
    
    for i in range(len(df)):
    
        if i>0:
        
            block1 = df.iloc[i]['Close']*2 / (1 + period)
            block2 = df.iloc[i-1]['Close'] * (1 - 2 / (1 + period))
    
            EMA.append(block1 + block2)
            
        else:
        
            EMA.append(float(df.iloc[0]['Close']))
            
    df['EMA'] = EMA

    return df


def add_k(df, period):
    """
    Stochastic Oscillator over period days.
    """

    df['L14'] = df.Low.rolling(period).min()
    df['H14'] = df.High.rolling(period).max()
    df['%K'] = ((df.Close - df.L14) / (df.H14 - df.L14)) * 100
    df.drop(columns=['L14', 'H14'], inplace=True)

    return df

def add_macd(df, period_fast, period_slow, period_signal):
    """
    Moving average convergence divergence.
    """
    
    MACD_fast = []
    
    for i in range(len(df)):
    
        if i > 0:
        
            block1 = df.iloc[i]['Close']*2 / (1 + period_fast)
            block2 = MACD_fast[i-1] * (1 - 2 / (1 + period_fast))
    
            MACD_fast.append(block1 + block2)
            
        else:
        
            MACD_fast.append(float(df.iloc[0]['Close']))
            
    MACD_slow = []
    
    for i in range(len(df)):
    
        if i > 0:
        
            block1 = df.iloc[i]['Close']*2 / (1 + period_slow)
            block2 = MACD_slow[i-1] * (1 - 2 / (1 + period_slow))
    
            MACD_slow.append(block1 + block2)
            
        else:
        
            MACD_slow.append(float(df.iloc[0]['Close']))
           
    MACD = np.array(MACD_fast) - np.array(MACD_slow)
    
    MACD_signal = []
    
    for i in range(len(df)):
    
        if i > 0:
        
            block1 = MACD[i]*2 / (1 + period_signal)
            block2 = MACD_signal[i-1] * (1 - 2 / (1 + period_signal))
    
            MACD_signal.append(block1 + block2)
            
        else:
        
            MACD_signal.append(float(MACD[0]))
            
    df['MACD'] = MACD_signal

    return df


def add_bb(df, period):
    """
    Bollinger Bands over period days.
    """

    df['30 Day MA'] = df['Adj Close'].rolling(window=period).mean()
    df['30 Day STD'] = df['Adj Close'].rolling(window=period).std()
    df['BB-Upper'] = df['30 Day MA'] + (df['30 Day STD'] * 2)
    df['BB-Lower'] = df['30 Day MA'] - (df['30 Day STD'] * 2)
    df.drop(columns=['30 Day MA', '30 Day STD'], inplace=True)

    return df


def add_rsi(df, period):
    """
    Relative Strength Index over period days.
    """

    close = df.Close
    delta = close.diff() 
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0
    
    # Calculate the exponential moving averages (EWMA)
    roll_up = up.ewm(com=period - 1, adjust=False).mean()
    roll_down = down.ewm(com=period - 1, adjust=False).mean().abs()
    
    # Calculate RS based on exponential moving average (EWMA)
    rs = roll_up / roll_down   # relative strength =  average gain/average loss

    rsi = 100-(100/(1+rs))
    df['RSI'] = rsi

    return df


def add_fibonacci(df, lookback_up, lookback_down):
    """
    Fibonacci retracement over lookback days.
    """

    df_ = df.copy()
    df_.index = range(len(df_))

    def minFunction(row):
        if row.name < lookback_down:
            return 0
        return df_['Close'].loc[row.name - lookback_down:row.name].min()

    def maxFunction(row):
        if row.name < lookback_up:
            return 0
        return df_['Close'].loc[row.name - lookback_up:row.name].max()

    df['Fibonacci_min'] = list(df_.apply(minFunction, axis=1))
    df['Fibonacci_max'] = list(df_.apply(maxFunction, axis=1))

    return df


def add_ichimoku(df):
    """
    Ichimoku cloud (one-look equilibrium chart).
    """

    df['tenkan_sen'] = (df['High'].rolling(window=9).max() +
                        df['Low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['High'].rolling(window=26).max() +
                       df['Low'].rolling(window=26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['High'].rolling(window=52).max(
    ) + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['Close'].shift(-22)

    return df


def add_std(df, period):
    """
    Standard deviation of closing price.
    """

    df['STD_CLOSE'] = df.Close.rolling(period).std()

    return df
    
def create_stock_dataframe(df):

    df_ = df.copy()
    df_ = StockDataFrame.retype(df_)

    return df_


def add_adx(df):
    """
    Average directional index.
    """

    df_ = create_stock_dataframe(df)
    df['ADX'] = df_['adx']

    return df


def add_r(df, period):
    """
    Calculate Larry William indicator (%R).
    """

    df['HH'] = df.High.rolling(period).max()
    df['LL'] = df.Low.rolling(period).min()
    df['%R'] = ((df.HH - df.Close) / (df.HH - df.LL)) * (-100)
    df.drop(columns=['HH', 'LL'], inplace=True)

    return df


def add_gainloss(df):
    """
    Net gain/loss of closing prices.
    """

    df['GAIN_LOSS'] = df['Close'].pct_change()

    return df
    
def add_inverted_hammer(df, invert=False):
    """
    Inverted Hammer.
    """
    name = 'Inverted Hammer'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df
    
def add_hammer(df, invert=False):
    """
    Hammer.
    """
    name = 'Hammer'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df
    
def add_hanging_man(df, invert=False):
    """
    Hanging Man.
    """
    name = 'Hanging Man'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df
    
def add_bearish_harami(df, invert=False):
    """
    Bearish Harami.
    """
    name = 'Bearish Harami'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df
    
def add_bullish_harami(df, invert=False):
    """
    Bullish Harami.
    """
    name = 'Bullish Harami'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df
    
def add_dark_cloud_cover(df, invert=False):
    """
    Dark Cloud Cover.
    """
    name = 'Dark Cloud Cover'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df
    
def add_doji(df, invert=False):
    """
    DOJI.
    """
    name = 'DOJI'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df
    
def add_doji_star(df, invert=False):
    """
    DOJI Star.
    """
    name = 'DOJI Star'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df
    
def add_dragonfly_doji(df, invert=False):
    """
    Dragonfly DOJI.
    """
    name = 'Dragonfly DOJI'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df   
    
def add_gravestone_doji(df, invert=False):
    """
    Gravestone DOJI.
    """
    name = 'Gravestone DOJI'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df  
    
def add_bearish_engulfing(df, invert=False):
    """
    Bearish Engulfing.
    """
    name = 'Bearish Engulfing'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df  
    
def add_bullish_engulfing(df, invert=False):
    """
    Bullish Engulfing.
    """
    name = 'Bullish Engulfing'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df  
    
def add_morning_star(df, invert=False):
    """
    Morning Star.
    """
    name = 'Morning Star'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df  
    
def add_morning_star_doji(df, invert=False):
    """
    Morning Star DOJI.
    """
    name = 'Morning Star DOJI'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df 
    
def add_piercing_pattern(df, invert=False):
    """
    Piercing.
    """
    name = 'Piercing'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df 
    
def add_rain_drop(df, invert=False):
    """
    Rain Drop.
    """
    name = 'Rain Drop'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df 
    
def add_rain_drop_doji(df, invert=False):
    """
    Rain Drop DOJI.
    """
    name = 'Rain Drop DOJI'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df 
    
def add_star(df, invert=False):
    """
    Star.
    """
    name = 'Star'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df 
    
def add_shooting_star(df, invert=False):
    """
    Shooting Star.
    """
    name = 'Shooting Star'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df 
    
def add_evening_star(df, invert=False):
    """
    Evening Star.
    """
    name = 'Evening Star'
    df = candlestick.evening_star(df, target=name)
    
    if invert:
        df[name] = df[name].replace({True: 1, False: 0})
    
    return df 


def get_historical_data(stock, years):
    """
    Retrieves historical data with financial features.
    """

    df, start, end = get_yahoo(stock, years)
    
    if 'Low' not in df.columns:
        df.Low = df.Open
    if 'High' not in df.columns:
        df.High = df.Close

    df = add_sma(df=df, period=20)
    df = add_ema(df=df, period=14)
    df = add_k(df=df, period=5)
    df = add_macd(df=df, period_fast=12, period_slow=26, period_signal=9)
    df = add_bb(df=df, period=14)
    df = add_rsi(df=df, period=14)
    df = add_fibonacci(df=df, lookback_up=235, lookback_down=135)
    df = add_ichimoku(df=df)
    df = add_std(df=df, period=14)
    df = add_adx(df=df)
    df = add_r(df=df, period=14)
    df = add_gainloss(df=df)
    
    df = add_inverted_hammer(df=df, invert=True)
    df = add_hammer(df=df, invert=True)
    df = add_hanging_man(df=df, invert=True)
    df = add_bearish_harami(df=df, invert=True)
    df = add_bullish_harami(df=df, invert=True)
    df = add_dark_cloud_cover(df=df, invert=True)
    df = add_doji(df=df, invert=True)
    df = add_doji_star(df=df, invert=True)
    df = add_dragonfly_doji(df=df, invert=True)
    df = add_gravestone_doji(df=df, invert=True)
    df = add_bearish_engulfing(df=df, invert=True)
    df = add_bullish_engulfing(df=df, invert=True)
    df = add_morning_star(df=df, invert=True)
    df = add_morning_star_doji(df=df, invert=True)
    df = add_piercing_pattern(df=df, invert=True)
    df = add_rain_drop(df=df, invert=True)
    df = add_rain_drop_doji(df=df, invert=True)
    df = add_star(df=df, invert=True)
    df = add_shooting_star(df=df, invert=True)
    df = add_evening_star(df=df, invert=True)
    
    print(df)
    
    assert 'Open' in df.columns
    assert 'High' in df.columns
    assert 'Low' in df.columns
    assert 'Close' in df.columns 

    print('STEP [1/3]: Data retrieved!')

    return df


if '__main__' == __name__:
    print('')
