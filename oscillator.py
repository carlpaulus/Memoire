# https://medium.com/codex/combining-bollinger-bands-and-stochastic-oscillator-to-create-a-killer-trading-strategy-in-python-6ea413a59037

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def get_historical_aapl(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)


aapl = get_historical_aapl('CS.PA', '2020-01-01', '2020-12-31')


# stochastic oscillator
def get_stoch_osc(high, low, close, k_lookback, d_lookback):
    lowest_low = low.rolling(k_lookback).min()
    highest_high = high.rolling(k_lookback).max()
    k_line = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    d_line = k_line.rolling(d_lookback).mean()
    return k_line, d_line


aapl['%k'], aapl['%d'] = get_stoch_osc(aapl['High'], aapl['Low'], aapl['Close'], 14, 3)
print(aapl.head(30))


# Creating the trading strategy
def implement_osc_strategy(prices, k, d):
    buy_price = []
    sell_price = []
    osc_signal = []
    signal = 0

    for i in range(len(prices)):
        if k[i] < 10 and d[i] < 10:  # k[i] < d[i] and  # k[i] > 80 and d[i] > 80: fait plus d'argent que la normal lol
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                osc_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                osc_signal.append(0)
        elif k[i] > 90 and d[i] > 90:  # k[i] > d[i] and  # k[i] < 20 and d[i] < 20: fait plus d'argent que la normal...
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                osc_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                osc_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            osc_signal.append(0)

    return buy_price, sell_price, osc_signal


buy_price, sell_price, rsi_signal = implement_osc_strategy(aapl['Close'], aapl['%k'], aapl['%d'])

ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
ax1.plot(aapl['Close'], linewidth=3, color='#ff9800', alpha=0.6, label='CS.PA')
ax1.set_title('AXA CLOSING PRICE', fontsize=14)
ax1.plot(aapl.index, buy_price, marker='^', color='#26a69a', markersize=12, linewidth=0, label='BUY SIGNAL')
ax1.plot(aapl.index, sell_price, marker='v', color='#f44336', markersize=12, linewidth=0, label='SELL SIGNAL')
ax1.legend(loc='center left', fontsize=10)
ax2.plot(aapl['%k'], color='#26a69a', label='Fast Stochastic', linewidth=3, alpha=0.3)
ax2.plot(aapl['%d'], color='#f44336', label='Slow Stochastic', linewidth=3, alpha=0.3)
ax2.axhline(10, color='grey', linewidth=2, linestyle='--')
ax2.axhline(90, color='grey', linewidth=2, linestyle='--')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_title('AXA Stochastic Oscillator', fontsize=14)
plt.show()


# Not the best I guess
# def stochastic_oscillator(data, slow=5, fast=15):
#     """ Stochastic (fast or slow) - overbought or oversold securities """
#     data['Lowest_5D'] = data['Low'].transform(lambda x: x.rolling(window=slow).min())
#     data['High_5D'] = data['High'].transform(lambda x: x.rolling(window=slow).max())
#     data['Lowest_15D'] = data['Low'].transform(lambda x: x.rolling(window=fast).min())
#     data['High_15D'] = data['High'].transform(lambda x: x.rolling(window=fast).max())
#
#     data['Stochastic_5'] = ((data['Close'] - data['Lowest_5D']) / (data['High_5D'] - data['Lowest_5D'])) * 100
#     data['Stochastic_15'] = ((data['Close'] - data['Lowest_15D']) / (data['High_15D'] - data['Lowest_15D'])) * 100
#
#     data['Stochastic_%D_5'] = data['Stochastic_5'].rolling(window=slow).mean()
#     data['Stochastic_%D_15'] = data['Stochastic_5'].rolling(window=fast).mean()
#
#     data['Stochastic_Ratio'] = data['Stochastic_%D_5'] / data['Stochastic_%D_15']
#     print(data['Stochastic_Ratio'])
#     return data['Stochastic_Ratio']
# stochastic_oscillator(aapl)


