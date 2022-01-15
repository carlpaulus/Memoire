# https://medium.com/codex/algorithmic-trading-with-williams-r-in-python-5a8e0db9ff1f

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')


def get_historical_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date).drop('Adj Close', axis=1)


nflx = get_historical_data('EL.PA', '2020-01-01', '2021-12-31')
# print(nflx)


def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max()
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr


nflx['wr_14'] = get_wr(nflx['High'], nflx['Low'], nflx['Close'], 14)
nflx = nflx.dropna()
print(nflx)

# ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
# ax1.plot(nflx['Close'], linewidth=2)
# ax1.set_title('NFLX CLOSING PRICE')
# ax2.plot(nflx['wr_14'], color='orange', linewidth=2)
# ax2.axhline(-20, linewidth=1.5, linestyle='--', color='grey')
# ax2.axhline(-80, linewidth=1.5, linestyle='--', color='grey')
# ax2.set_title('NFLX WILLIAMS %R 14')
# plt.show()


def implement_wr_strategy(prices, wr):
    buy_price = []
    sell_price = []
    wr_signal = []
    signal = 0

    for i in range(len(wr)):
        if wr[i - 1] > -80 and wr[i] < -80:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                wr_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                wr_signal.append(0)
        elif wr[i - 1] < -20 and wr[i] > -20:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                wr_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                wr_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            wr_signal.append(0)

    return buy_price, sell_price, wr_signal


buy_price, sell_price, wr_signal = implement_wr_strategy(nflx['Close'], nflx['wr_14'])

#  plotting the trading signals
ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
ax1.plot(nflx['Close'], linewidth=2, label='EL.PA')
ax1.plot(nflx.index, buy_price, marker='^', markersize=10, linewidth=0, color='green', label='BUY SIGNAL')
ax1.plot(nflx.index, sell_price, marker='v', markersize=10, linewidth=0, color='r', label='SELL SIGNAL')
ax1.legend(loc='upper left', fontsize=12)
ax1.set_title('ESSILOR TRADING SIGNALS')
ax2.plot(nflx['wr_14'], color='orange', linewidth=2)
ax2.axhline(-20, linewidth=1.5, linestyle='--', color='grey')
ax2.axhline(-80, linewidth=1.5, linestyle='--', color='grey')
ax2.set_title('ESSILOR WILLIAMS %R 14')
plt.show()

# creating our positions
position = []
for i in range(len(wr_signal)):
    if wr_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)

for i in range(len(nflx['Close'])):
    if wr_signal[i] == 1:
        position[i] = 1
    elif wr_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i - 1]

close_price = nflx['Close']
wr = nflx['wr_14']
wr_signal = pd.DataFrame(wr_signal).rename(columns={0: 'wr_signal'}).set_index(nflx.index)
position = pd.DataFrame(position).rename(columns={0: 'wr_position'}).set_index(nflx.index)

frames = [close_price, wr, wr_signal, position]
strategy = pd.concat(frames, join='inner', axis=1)

print(strategy)

# backtesting
nflx_ret = pd.DataFrame(np.diff(nflx['Close'])).rename(columns={0: 'returns'})
wr_strategy_ret = []

for i in range(len(nflx_ret)):
    returns = nflx_ret['returns'][i] * strategy['wr_position'][i]
    wr_strategy_ret.append(returns)

wr_strategy_ret_df = pd.DataFrame(wr_strategy_ret).rename(columns={0: 'wr_returns'})
investment_value = 100000
number_of_stocks = floor(investment_value / nflx['Close'][-1])
wr_investment_ret = []

for i in range(len(wr_strategy_ret_df['wr_returns'])):
    returns = number_of_stocks * wr_strategy_ret_df['wr_returns'][i]
    wr_investment_ret.append(returns)

wr_investment_ret_df = pd.DataFrame(wr_investment_ret).rename(columns={0: 'investment_returns'})
total_investment_ret = round(sum(wr_investment_ret_df['investment_returns']), 2)
profit_percentage = floor((total_investment_ret / investment_value) * 100)
print(cl('Profit gained from the W%R strategy by investing $100k in NFLX : {}'.format(total_investment_ret),
         attrs=['bold']))
print(cl('Profit percentage of the W%R strategy : {}%'.format(profit_percentage), attrs=['bold']))


# SPY ETF comparison
def get_benchmark(investment_value, start_date, end_date):
    spy = get_historical_data('ENR.DE', start_date, end_date)['Close']
    benchmark = pd.DataFrame(np.diff(spy)).rename(columns={0: 'benchmark_returns'})

    investment_value = investment_value
    number_of_stocks = floor(investment_value / spy[-1])
    benchmark_investment_ret = []

    for i in range(len(benchmark['benchmark_returns'])):
        returns = number_of_stocks * benchmark['benchmark_returns'][i]
        benchmark_investment_ret.append(returns)

    benchmark_investment_ret_df = pd.DataFrame(benchmark_investment_ret).rename(columns={0: 'investment_returns'})
    return benchmark_investment_ret_df


benchmark = get_benchmark(100000, '2020-01-01', '2020-12-31')

investment_value = 100000
total_benchmark_investment_ret = round(sum(benchmark['investment_returns']), 2)
benchmark_profit_percentage = floor((total_benchmark_investment_ret / investment_value) * 100)
print(cl('Benchmark profit by investing $100k : {}'.format(total_benchmark_investment_ret), attrs=['bold']))
print(cl('Benchmark Profit percentage : {}%'.format(benchmark_profit_percentage), attrs=['bold']))
print(cl('W%R Strategy profit is {}% higher than the Benchmark Profit'.format(
      profit_percentage - benchmark_profit_percentage), attrs=['bold']))
