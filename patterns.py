import pandas as pd
from pandas.api.types import is_numeric_dtype

class CandlestickFinder(object):
    def __init__(self, name, required_count, target=None):
        self.name = name
        self.required_count = required_count
        self.close_column = 'Close'
        self.open_column = 'Open'
        self.low_column = 'Low'
        self.high_column = 'High'
        self.data = None
        self.is_data_prepared = False
        self.multi_coeff = -1

        if target:
            self.target = target
        else:
            self.target = self.name

    def get_class_name(self):
        return self.__class__.__name__

    def logic(self, row_idx):
        raise Exception('Implement the logic of ' + self.get_class_name())

    def has_pattern(self,
                    candles_df,
                    ohlc,
                    is_reversed):
        self.prepare_data(candles_df,
                          ohlc)

        if self.is_data_prepared:
            results = []
            rows_len = len(candles_df)
            idxs = candles_df.index.values

            if is_reversed:
                self.multi_coeff = 1

                for row_idx in range(rows_len - 1, -1, -1):

                    if row_idx <= rows_len - self.required_count:
                        results.append([idxs[row_idx], self.logic(row_idx)])
                    else:
                        results.append([idxs[row_idx], None])

            else:
                self.multi_coeff = -1

                for row in range(0, rows_len, 1):

                    if row >= self.required_count - 1:
                        results.append([idxs[row], self.logic(row)])
                    else:
                        results.append([idxs[row], None])

            candles_df = candles_df.join(pd.DataFrame(results, columns=['row', self.target]).set_index('row'),
                                         how='outer')

            return candles_df
        else:
            raise Exception('Data is not prepared to detect patterns')

    def prepare_data(self, candles_df, ohlc):

        if isinstance(candles_df, pd.DataFrame):

            if len(candles_df) >= self.required_count:
                if ohlc and len(ohlc) == 4:
                    if not set(ohlc).issubset(candles_df.columns):
                        raise Exception('Provided columns does not exist in given data frame')

                    self.open_column = ohlc[0]
                    self.high_column = ohlc[1]
                    self.low_column = ohlc[2]
                    self.close_column = ohlc[3]
                else:
                    raise Exception('Provide list of four elements indicating columns in strings. '
                                    'Default: [open, high, low, close]')

                self.data = candles_df.copy()

                if not is_numeric_dtype(self.data[self.close_column]):
                    self.data[self.close_column] = pd.to_numeric(self.data[self.close_column])

                if not is_numeric_dtype(self.data[self.open_column]):
                    self.data[self.open_column] = pd.to_numeric(self.data[self.open_column])

                if not is_numeric_dtype(self.data[self.low_column]):
                    self.data[self.low_column] = pd.to_numeric(self.data[self.low_column])

                if not is_numeric_dtype(self.data[self.high_column]):
                    self.data[self.high_column] = pd.to_numeric(candles_df[self.high_column])

                self.is_data_prepared = True
            else:
                raise Exception('{0} requires at least {1} data'.format(self.name,
                                                                        self.required_count))
        else:
            raise Exception('Candles must be in Panda data frame type')


class BearishEngulfing(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]
          
        return (open >= prev_close > prev_open and
                open > close and
                prev_open >= close and 
                open - close > prev_close - prev_open)
        
        # return (prev_close > prev_open and
        #         0.3 > abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.1 and
        #         close < open and
        #         abs(close - open) / (high - low) >= 0.7 and
        #         prev_high < open and
        #         prev_low > close)
        
class BearishHarami(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        # return (prev_close > prev_open and
        #        abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7 and
        #        0.3 > abs(close - open) / (high - low) >= 0.1 and
        #        high < prev_close and
        #        low > prev_open)

        return (prev_close > prev_open and
                prev_open <= close < open <= prev_close and
                open - close < prev_close - prev_open)
                
class BullishEngulfing(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        # return (prev_close < prev_open and
        #         0.3 > abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.1 and
        #         close > open and
        #         abs(close - open) / (high - low) >= 0.7 and
        #         prev_high < close and
        #         prev_low > open)

        return (close >= prev_open > prev_close and
                close > open and
                prev_close >= open and
                close - open > prev_open - prev_close)
                
class BullishHarami(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        # return (prev_close < prev_open and
        #        abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7
        #        and 0.3 > abs(close - open) / (high - low) >= 0.1
        #        and high < prev_open
        #        and low > prev_close)
        
        return (prev_open > prev_close and
                prev_close <= open < close <= prev_open and
                close - open < prev_open - prev_close)
                
class DarkCloudCover(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        # return prev_close > prev_open and \
        #        abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7 and \
        #        close < open and \
        #        abs(close - open) / (high - low) >= 0.7 and \
        #        open >= prev_close and \
        #        prev_open < close < (prev_open + prev_close) / 2

        return ((prev_close > prev_open) and
                (((prev_close + prev_open) / 2) > close) and
                (open > close) and
                (open > prev_close) and
                (close > prev_open) and
                ((open - close) / (.001 + (high - low)) > 0.6))
                
class Doji(CandlestickFinder):
    
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 1, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        return abs(close - open) / (high - low) < 0.1 and \
               (high - max(close, open)) > (3 * abs(close - open)) and \
               (min(close, open) - low) > (3 * abs(close - open))
              
class DojiStar(CandlestickFinder):

    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        return prev_close > prev_open and \
               abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7 and \
               abs(close - open) / (high - low) < 0.1 and \
               prev_close < close and \
               prev_close < open and \
               (high - max(close, open)) > (3 * abs(close - open)) and \
               (min(close, open) - low) > (3 * abs(close - open))
               
class DragonflyDoji(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 1, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        return abs(close - open) / (high - low) < 0.1 and \
               (min(close, open) - low) > (3 * abs(close - open)) and \
               (high - max(close, open)) < abs(close - open)
               
class EveningStar(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 3, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]
        b_prev_candle = self.data.iloc[idx + 2 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        b_prev_close = b_prev_candle[self.close_column]
        b_prev_open = b_prev_candle[self.open_column]
        b_prev_high = b_prev_candle[self.high_column]
        b_prev_low = b_prev_candle[self.low_column]
        

        # return (b_prev_close > b_prev_open and
        #         abs(b_prev_close - b_prev_open) / (b_prev_high - b_prev_low) >= 0.7 and
        #         0.3 > abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.1 and
        #         close < open and
        #         abs(close - open) / (high - low) >= 0.7 and
        #         b_prev_close < prev_close and
        #         b_prev_close < prev_open and
        #         prev_close > open and
        #         prev_open > open and
        #         close < b_prev_close)

        return (min(prev_open, prev_close) > b_prev_close > b_prev_open and
                close < open < min(prev_open, prev_close))
            
                
class GravestoneDoji(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 1, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        return (abs(close - open) / (high - low) < 0.1 and
                (high - max(close, open)) > (3 * abs(close - open)) and
                (min(close, open) - low) <= abs(close - open))
                
class Hammer(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 1, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        return (((high - low) > 3 * (open - close)) and
                ((close - low) / (.001 + high - low) > 0.6) and
                ((open - low) / (.001 + high - low) > 0.6))
                
class HangingMan(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 3, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]
        b_prev_candle = self.data.iloc[idx + 2 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        b_prev_close = b_prev_candle[self.close_column]
        b_prev_open = b_prev_candle[self.open_column]
        b_prev_high = b_prev_candle[self.high_column]
        b_prev_low = b_prev_candle[self.low_column]

        # return (((high - low > 4 * (open - close)) and
        #          ((close - low) / (.001 + high - low) >= 0.75) and
        #          ((open - low) / (.001 + high - low) >= 0.75)) and
        #         high[1] < open and
        #         high[2] < open)

        return (((high - low > 4 * (open - close)) and
                 ((close - low) / (.001 + high - low) >= 0.75) and
                 ((open - low) / (.001 + high - low) >= 0.75)) and
                prev_high < open and
                b_prev_high < open)
                
class InvertedHammer(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 1, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        return (((high - low) > 3 * (open - close)) and
                ((high - close) / (.001 + high - low) > 0.6)
                and ((high - open) / (.001 + high - low) > 0.6))
                
class MorningStar(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 3, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]
        b_prev_candle = self.data.iloc[idx + 2 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        b_prev_close = b_prev_candle[self.close_column]
        b_prev_open = b_prev_candle[self.open_column]
        b_prev_high = b_prev_candle[self.high_column]
        b_prev_low = b_prev_candle[self.low_column]

        # return (b_prev_close < b_prev_open and
        #         abs(b_prev_close - b_prev_open) / (b_prev_high - b_prev_low) >= 0.7 and
        #         0.3 > abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.1 and
        #         close > open and
        #         abs(close - open) / (high - low) >= 0.7 and
        #         b_prev_close > prev_close and
        #         b_prev_close > prev_open and
        #         prev_close < open and
        #         prev_open < open and
        #         close > b_prev_close)

        return (max(prev_open, prev_close) < b_prev_close < b_prev_open and
                close > open > max(prev_open, prev_close))
                
class MorningStarDoji(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 3, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]
        b_prev_candle = self.data.iloc[idx + 2 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        b_prev_close = b_prev_candle[self.close_column]
        b_prev_open = b_prev_candle[self.open_column]
        b_prev_high = b_prev_candle[self.high_column]
        b_prev_low = b_prev_candle[self.low_column]

        return (b_prev_close < b_prev_open and
                abs(b_prev_close - b_prev_open) / (b_prev_high - b_prev_low) >= 0.7 and
                abs(prev_close - prev_open) / (prev_high - prev_low) < 0.1 and
                close > open and
                abs(close - open) / (high - low) >= 0.7 and
                b_prev_close > prev_close and
                b_prev_close > prev_open and
                prev_close < open and
                prev_open < open and
                close > b_prev_close
                and (prev_high - max(prev_close, prev_open)) > (3 * abs(prev_close - prev_open))
                and (min(prev_close, prev_open) - prev_low) > (3 * abs(prev_close - prev_open)))
                
class PiercingPattern(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        # return (prev_close < prev_open and
        #         abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7 and
        #         close > open and
        #         abs(close - open) / (high - low) >= 0.7 and
        #         open <= prev_close and
        #         close < prev_open and
        #         close < ((prev_open + prev_close) / 2))

        return (prev_close < prev_open and
                open < prev_low and
                prev_open > close > prev_close + ((prev_open - prev_close) / 2))
                
class RainDrop(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        return (prev_close < prev_open and
                abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7 and
                0.3 > abs(close - open) / (high - low) >= 0.1 and
                prev_close > close and
                prev_close > open)
                
class RainDropDoji(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        return (prev_close < prev_open and
                abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7 and
                abs(close - open) / (high - low) < 0.1 and
                prev_close > close and
                prev_close > open and
                (high - max(close, open)) > (3 * abs(close - open)) and
                (min(close, open) - low) > (3 * abs(close - open)))
                
class ShootingStar(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        return (prev_open < prev_close < open and
                high - max(open, close) >= abs(open - close) * 3 and
                min(close, open) - low <= abs(open - close))
                
class Star(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        return (prev_close > prev_open and
                abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7 and
                0.3 > abs(close - open) / (high - low) >= 0.1 and
                prev_close < close and
                prev_close < open)
