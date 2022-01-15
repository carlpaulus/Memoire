# https://medium.com/codex/algorithmic-trading-with-sma-in-python-7d66008d37b1
# https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9
# https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import math
from termcolor import colored as cl
import numpy as np

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)


# EXTRACTING DATA

def get_historic_data(symbol):
    return yf.download(symbol, start='2020-02-01', end='2021-02-01').drop('Adj Close', axis=1)  # .drop('Volume', axis=1)


msft = get_historic_data('MC.PA')
# print(msft)


# DEFINING SMA/EMA/WMA FUNCTION
def sma(data, n):
    # sma = data.rolling(window=n).mean()
    ema = data.ewm(span=n, adjust=False).mean()
    # weights = np.arange(1,i+1)
    # wma = data.rolling(n).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    return pd.DataFrame(ema)


n = [20, 50]
for i in n:
    msft[f'sma_{i}'] = sma(msft['Volume'], i)

# PLOTTING SMA VALUES

# plt.plot(msft['Volume'], label='MSFT', linewidth=5, alpha=0.3)
# plt.plot(msft['sma_20'], label='SMA 20')
# plt.plot(msft['sma_50'], label='SMA 50')
# plt.title('MSFT Simple Moving Averages (20, 50)')
# plt.legend(loc='upper left')
# plt.show()


# CREATING SMA TRADING STRATEGY

def implement_sma_strategy(data, short_window, long_window):
    sma1 = short_window
    sma2 = long_window
    buy_price = []
    sell_price = []
    sma_signal = []
    signal = 0

    for i in range(len(data)):
        if sma1[i] > sma2[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                sma_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                sma_signal.append(0)
        elif sma2[i] > sma1[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                sma_signal.append(-1)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                sma_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            sma_signal.append(0)

    return buy_price, sell_price, sma_signal


sma_20 = msft['sma_20']
sma_50 = msft['sma_50']

buy_price, sell_price, signal = implement_sma_strategy(msft['Close'], sma_20, sma_50)

# PLOTTING SMA TRADE SIGNALS
ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)

ax1.plot(msft['Close'], alpha=0.3, label='MC.PA')
ax1.plot(msft.index, buy_price, marker='^', markersize=12, linewidth=0, color='darkblue', label='BUY SIGNAL')
ax1.plot(msft.index, sell_price, marker='v', markersize=12, linewidth=0, color='crimson', label='SELL SIGNAL')
ax1.legend(loc='upper left', fontsize=12)
ax1.set_title('LVMH EMA TRADING SIGNALS')

ax2.plot(sma_20, alpha=0.6, label='SMA 20')
ax2.plot(sma_50, alpha=0.6, label='SMA 50')
ax2.legend(loc='upper left', fontsize=12)
ax2.set_title('LVMH EMA VOLUME CROSSOVER')
plt.show()

# OUR POSITION IN STOCK (HOLD/SOLD)

position = []
for i in range(len(signal)):
    if signal[i] > 1:
        position.append(0)
    else:
        position.append(1)

for i in range(len(msft['Volume'])):
    if signal[i] == 1:
        position[i] = 1
    elif signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i - 1]

# CONSOLIDATING LISTS TO DATAFRAME

sma_20 = pd.DataFrame(sma_20).rename(columns={0: 'sma_20'})
sma_50 = pd.DataFrame(sma_50).rename(columns={0: 'sma_50'})
buy_price = pd.DataFrame(buy_price).rename(columns={0: 'buy_price'}).set_index(msft.index)
sell_price = pd.DataFrame(sell_price).rename(columns={0: 'sell_price'}).set_index(msft.index)
signal = pd.DataFrame(signal).rename(columns={0: 'sma_signal'}).set_index(msft.index)
position = pd.DataFrame(position).rename(columns={0: 'sma_position'}).set_index(msft.index)

frames = [sma_20, sma_50, buy_price, sell_price, signal, position]
strategy = pd.concat(frames, join='inner', axis=1)
strategy = strategy.reset_index().drop('Date', axis=1)
# print(strategy)

# BACKTESTING THE STRAGEGY

msft_ret = pd.DataFrame(np.diff(msft['Close'])).rename(columns={0: 'returns'})
#print(msft_ret)
sma_strategy_ret = []

for i in range(len(msft_ret)):
    try:
        returns = msft_ret['returns'][i] * strategy['sma_position'][i]
        sma_strategy_ret.append(returns)
    except:
        pass

sma_strategy_ret_df = pd.DataFrame(sma_strategy_ret).rename(columns={0: 'sma_returns'})

investment_value = 100000
number_of_stocks = math.floor(investment_value / msft['Close'][1])
sma_investment_ret = []

for i in range(len(sma_strategy_ret_df['sma_returns'])):
    returns = number_of_stocks * sma_strategy_ret_df['sma_returns'][i]
    sma_investment_ret.append(returns)

sma_investment_ret_df = pd.DataFrame(sma_investment_ret).rename(columns={0: 'investment_returns'})
total_investment_ret = round(sum(sma_investment_ret_df['investment_returns']), 2)
print(cl('Profit gained from the strategy by investing $100K in MSFT : ${} in 1 Year'.format(total_investment_ret),
         attrs=['bold']))
