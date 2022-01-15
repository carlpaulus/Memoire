from portfolios import PortfolioConstruction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


class FactorConstructions(PortfolioConstruction):

    def __init__(self):
        super().__init__()
        self.get_benchmark_data()
        self.get_benchmark_level()
        self.get_benchmark_volumes()
        self.get_benchmark_highs()
        self.get_benchmark_lows()

        if self.market == 'europe':
            self.factors = "Europe_5_Factors_Daily.csv"
            self.carhart_momentum = "Europe_MOM_Factor_Daily.csv"

        elif self.market == 'america':
            self.factors = "North_America_5_Factors_Daily.csv"
            self.carhart_momentum = "North_America_MOM_Factor_Daily.csv"

        elif self.market == 'japan':
            self.factors = "Japan_5_Factors_Daily.csv"
            self.carhart_momentum = "Japan_MOM_Factor_Daily.csv"

        else:
            AttributeError('for which market do you need FF factors?')

    @staticmethod
    def normalize(x: pd.Series):
        return (x - x.mean()) / x.std()

    """ Fama & French factors """

    def get_famafrench_factors(self):
        _raw = pd.read_csv(self.factors, index_col=0, parse_dates=True, skiprows=5).truncate(before=self.start,
                                                                                             after=self.end)
        if self.component_log_returns is None:
            raise TypeError("self.component_log_returns is None")
        index_ = _raw[_raw.index.isin(self.component_log_returns.index)]
        _ = index_.pop("RF")
        return index_

    def get_carhart_momentum(self):
        _raw = pd.read_csv(self.carhart_momentum, index_col=0, parse_dates=True, skiprows=5).truncate(before=self.start,
                                                                                                      after=self.end)
        if self.component_log_returns is None:
            raise TypeError("self.component_log_returns is None")
        index_ = _raw[_raw.index.isin(self.component_log_returns.index)]
        return index_

    """ Trend following factors Moving Averages: can be run on prices and volumes """

    # Simple Moving Average
    def sma(self, criteria='Close', slow=14, fast=3):  # the criteria can also be 'Volume'
        slow_table = pd.DataFrame(self.benchmark_data[criteria].rolling(window=slow).mean()).rename(
            columns={criteria: criteria + '_slowSMA'})
        fast_table = pd.DataFrame(self.benchmark_data[criteria].rolling(window=fast).mean()).rename(
            columns={criteria: criteria + '_fastSMA'})
        table = pd.concat([slow_table, fast_table], join='inner', axis=1)

        # Create the time series of the factor
        time_series = pd.DataFrame(table[criteria + '_fastSMA'])
        return time_series.apply(self.normalize, axis=0)[criteria + '_fastSMA'].fillna(0).rename(criteria + '_SMA')

    # Exponential Moving Average
    def ema(self, criteria='Close', slow=14, fast=3):  # criteria can be 'Volume' also
        slow_table = pd.DataFrame(self.benchmark_data[criteria].ewm(span=slow, adjust=False).mean()).rename(
            columns={criteria: criteria + '_slowEMA'})
        fast_table = pd.DataFrame(self.benchmark_data[criteria].ewm(span=fast, adjust=False).mean()).rename(
            columns={criteria: criteria + '_fastEMA'})
        table = pd.concat([slow_table, fast_table], join='inner', axis=1)

        # Create the time series of the factor
        time_series = pd.DataFrame(table[criteria + '_fastEMA'] - table[criteria + '_slowEMA'])
        return time_series.apply(self.normalize, axis=0)[0].fillna(0).rename(criteria + '_EMA')

    # Weighted Moving Average
    def wma(self, criteria='Close', slow=14, fast=3):  # criteria can be 'Volume' also
        slow_weights = np.arange(1, slow+1)
        slow_table = self.benchmark_data[criteria].rolling(window=slow).apply(
            lambda prices: np.dot(prices, slow_weights) / slow_weights.sum(), raw=True)

        fast_weights = np.arange(1, fast+1)
        fast_table = self.benchmark_data[criteria].rolling(window=fast).apply(
            lambda prices: np.dot(prices, fast_weights) / fast_weights.sum(), raw=True)

        all_wma = pd.concat([slow_table, fast_table], join='inner', axis=1)
        all_wma.columns = [criteria + '_slowWMA', criteria + '_fastWMA']

        # Create the time series of the factor
        time_series = pd.DataFrame(all_wma[criteria + '_fastWMA'])
        return time_series.apply(self.normalize, axis=0)[criteria + '_fastWMA'].fillna(0).rename(criteria + '_WMA')

    def macd(self, criteria='Close', slow=21, fast=9, smooth=9):
        exp1 = self.benchmark_data[criteria].ewm(span=fast, adjust=False).mean()
        exp2 = self.benchmark_data[criteria].ewm(span=slow, adjust=False).mean()
        macd = pd.DataFrame(exp1 - exp2).rename(columns={criteria: criteria + '_macd'})
        signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(columns={criteria + '_macd': 'signal'})
        hist = pd.DataFrame(macd[criteria + '_macd'] - signal['signal']).rename(columns={0: 'hist'})
        all_macd = pd.concat([macd, signal, hist], join='inner', axis=1)

        # Create the time series of the factor
        return all_macd.apply(self.normalize, axis=0)[criteria + '_macd'].fillna(0).rename(criteria + '_macd')

    """ Oscillating factors: stochastic oscillator, rsi, w%r, obv, dmi """

    def oscillator(self, k_lookback=14, d_lookback=3):
        high = self.benchmark_high
        low = self.benchmark_low

        lowest_low = low.rolling(k_lookback).min()
        highest_high = high.rolling(k_lookback).max()
        k_line = ((self.benchmark_prices - lowest_low) / (highest_high - lowest_low)) * 100
        d_line = k_line.rolling(d_lookback).mean()
        all_osc = pd.concat([k_line, d_line], join='inner', axis=1).rename(columns={0: 'k_line', 1: 'd_line'})

        # Create the time series of the factor
        return all_osc.apply(self.normalize, axis=0)['k_line'].fillna(0).rename('OSC')

    def rsi(self, lookback=14):
        ret = self.benchmark_prices.diff()
        up = []
        down = []
        for i in range(len(ret)):
            if ret[i] < 0:
                up.append(0)
                down.append(ret[i])
            else:
                up.append(ret[i])
                down.append(0)
        up_series = pd.Series(up)
        down_series = pd.Series(down).abs()
        up_ewm = up_series.ewm(com=lookback - 1, adjust=False).mean()
        down_ewm = down_series.ewm(com=lookback - 1, adjust=False).mean()
        rs = up_ewm / down_ewm
        rsi = 100 - (100 / (1 + rs))
        rsi_df = pd.DataFrame(rsi).rename(columns={0: 'rsi'}).set_index(self.benchmark_prices.index)
        rsi_df = rsi_df.dropna()

        # Create the time series of the factor
        return rsi_df.apply(self.normalize, axis=0)['rsi'].fillna(0).rename('RSI')

    def william_r(self, lookback=14):
        high = self.benchmark_high
        low = self.benchmark_low

        highh = high.rolling(lookback).max()
        lowl = low.rolling(lookback).min()
        wr = -100 * ((highh - self.benchmark_prices) / (highh - lowl))

        # Create the time series of the factor
        return pd.DataFrame(wr).apply(self.normalize, axis=0)[0].fillna(0).rename('WR')

    def on_balance_volume(self):  # and obv_ema
        prices = self.benchmark_prices
        volumes = self.benchmark_volumes

        obv = [0]
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif prices[i] < prices[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])

        obv = pd.Series(obv, index=prices.index)
        obv_ema = obv.ewm(span=20).mean()

        # Create the time series of the factor
        return pd.DataFrame(obv_ema).apply(self.normalize, axis=0)[0].fillna(0).rename('OBV')

    def dmi(self, lookback=14):
        high = self.benchmark_high
        low = self.benchmark_low

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - self.benchmark_prices.shift(1)))
        tr3 = pd.DataFrame(abs(low - self.benchmark_prices.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(lookback).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1 / lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1 / lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        # adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        # adx_smooth = adx.ewm(alpha=1 / lookback).mean()

        # Create the time series of the factor
        return pd.DataFrame(dx).apply(self.normalize, axis=0)[0].fillna(0).rename('DMI')

    """ Volatility factors: bollinger bands, fibonacci"""

    def bollinger_bands(self, lookback=14):
        rolling_returns = self.benchmark_prices.rolling(window=lookback)
        sma = rolling_returns.mean()
        std = rolling_returns.std()
        # upper_bb = sma + std * 2
        lower_bb = sma - std * 2

        # Create the time series of the factor
        time_series = pd.DataFrame(lower_bb)  # upper_bb - lower_bb
        return time_series.apply(self.normalize, axis=0)['Close'].fillna(0).rename('BB')

    def fibonacci(self):
        data = self.benchmark_data
        ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

        period_length = 20
        fibonacci_levels = [[np.nan] * len(ratios)] * period_length

        for i in range(period_length, len(data)):

            highest_swing = -1
            lowest_swing = -1

            daily_df = data[i-period_length:i]
            df_high = daily_df['High']
            df_low = daily_df['Low']

            for j in range(1, df_high.shape[0] - 1):
                if df_high[j] > df_high[j - 1] and df_high[j] > df_high[j + 1] and (
                        highest_swing == -1 or df_high[j] > df_high[highest_swing]):
                    highest_swing = j

                if df_low[j] < df_low[j - 1] and df_low[j] < df_low[j + 1] and (
                        lowest_swing == -1 or df_low[j] < df_low[lowest_swing]):
                    lowest_swing = j

            daily_levels = []
            max_level = df_high[highest_swing]
            min_level = df_low[lowest_swing]

            for ratio in ratios:
                if highest_swing > lowest_swing:  # Uptrend
                    daily_levels.append(max_level - (max_level - min_level) * ratio)  # ratio*100 ?
                else:  # Downtrend
                    daily_levels.append(min_level + (max_level - min_level) * ratio)  # ratio*100 ?

            fibonacci_levels.append(daily_levels)

        # Create the time series of the factor
        time_series = pd.DataFrame(fibonacci_levels, columns=ratios, index=data.index)
        return time_series.apply(self.normalize, axis=0)[0.618].fillna(0).rename('Fibo')


# calculate factor combinations (by 2) + Momentum, RMW, CMA
class Combinations(FactorConstructions):

    def __init__(self):
        super().__init__()
        self.three_factors = self.get_famafrench_factors().drop(['RMW', 'CMA'], axis=1)

        # FF-based factors
        self.rmw = self.get_famafrench_factors()["RMW"]
        self.cma = self.get_famafrench_factors()["CMA"]
        self.wml = self.get_carhart_momentum()["WML"]  # mom

        # tailored trending factors
        self.sma_price = self.sma('Close')
        self.ema_price = self.ema('Close')
        self.wma_price = self.wma('Close')
        self.macd_price = self.macd('Close')

        self.sma_volume = self.sma('Volume')
        self.ema_volume = self.ema('Volume')
        self.wma_volume = self.wma('Volume')
        self.macd_volume = self.macd('Volume')

        # tailored oscillator factors
        self.osc = self.oscillator()
        self.rsi = self.rsi()
        self.wr = self.william_r()
        self.dmi = self.dmi()
        self.obv = self.on_balance_volume()

        # tailored volatility factors
        self.bb = self.bollinger_bands()
        self.fib = self.fibonacci()

    def five_factor_combinations(self):
        factor_df = []
        factor_list = [self.rmw, self.cma, self.wml, self.sma_price, self.sma_volume, self.ema_price,
                       self.ema_volume, self.wma_price, self.wma_volume, self.macd_price, self.macd_volume,
                       self.osc, self.rsi, self.wr, self.obv, self.dmi, self.bb, self.fib]  # self.w_volume

        factor_list_copy = factor_list.copy()

        couples = []
        dust_bin = []

        for factor in factor_list:
            for factor_copy in factor_list_copy:
                if factor.name != factor_copy.name and factor_copy.name not in dust_bin:
                    couples.append((factor.name, factor_copy.name))
                    factor_df.append(pd.concat([self.three_factors, factor, factor_copy], join='inner', axis=1))
            dust_bin.append(factor.name)

        print(len(couples))
        return factor_df


if __name__ == '__main__':
    FC = FactorConstructions()
    print(FC.wma())
    # print(Combinations().five_factor_combinations())




