import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import math
from termcolor import colored as cl
import numpy as np

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)


def get_historic_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date).drop('Adj Close', axis=1).drop('Volume', axis=1)


tsla = get_historic_data('AI.PA', '2020-01-01', '2021-12-31')
# print(tsla.tail())

# plt.plot(tsla.index, tsla['Close'])
# plt.xlabel('Date')
# plt.ylabel('Closing Prices')
# plt.title('TSLA Stock Prices 2020-2021')
# plt.show()


def sma(data, window):
    sma = data.rolling(window=window).mean()
    return sma


tsla['sma_20'] = sma(tsla['Close'], 30)
# print(tsla.tail())

# tsla['Close'].plot(label='CLOSE', alpha=0.6)
# tsla['sma_20'].plot(label='SMA 20', linewidth=2)
# plt.xlabel('Date')
# plt.ylabel('Closing Prices')
# plt.legend(loc='upper left')
# plt.show()


def bb(data, sma, window):
    std = data.rolling(window=window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb


tsla['upper_bb'], tsla['lower_bb'] = bb(tsla['Close'], tsla['sma_20'], 30)
print(tsla.tail())

""" other way to calculate the BB are 
# BOLLINGER BANDS CALCULATION

def sma(data, lookback):
    sma = data.rolling(lookback).mean()
    return sma

def get_bb(data, lookback):
    std = data.rolling(lookback).std()
    upper_bb = sma(data, lookback) + std * 2
    lower_bb = sma(data, lookback) - std * 2
    middle_bb = sma(data, lookback)
    return upper_bb, lower_bb, middle_bb

aapl['upper_bb'], aapl['middle_bb'], aapl['lower_bb'] = get_bb(aapl['close'], 20)
aapl = aapl.dropna()
aapl.tail()
"""

# tsla['Close'].plot(label='CLOSE PRICES', color='skyblue')
# tsla['upper_bb'].plot(label='UPPER BB 20', linestyle='--', linewidth=1, color='black')
# tsla['sma_20'].plot(label='MIDDLE BB 20', linestyle='--', linewidth=1.2, color='grey')
# tsla['lower_bb'].plot(label='LOWER BB 20', linestyle='--', linewidth=1, color='black')
# plt.legend(loc='upper left')
# plt.title('TSLA BOLLINGER BANDS')
# plt.show()


def implement_bb_strategy(data, lower_bb, upper_bb):
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0

    for i in range(len(data)):
        if data[i - 1] > lower_bb[i - 1] and data[i] < lower_bb[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        elif data[i - 1] < upper_bb[i - 1] and data[i] > upper_bb[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)

    return buy_price, sell_price, bb_signal


buy_price, sell_price, bb_signal = implement_bb_strategy(tsla['Close'], tsla['lower_bb'], tsla['upper_bb'])

tsla['Close'].plot(label='AI.PA', alpha=0.3)
tsla['upper_bb'].plot(label='UPPER BB', linestyle='--', linewidth=1, color='black')
tsla['sma_20'].plot(label='SMA 30', linestyle='--', linewidth=1.2, color='grey')
tsla['lower_bb'].plot(label='LOWER BB', linestyle='--', linewidth=1, color='black')
plt.plot(tsla.index, buy_price, marker='^', color='green', markersize=12, label='BUY SIGNAL')
plt.plot(tsla.index, sell_price, marker='v', color='red', markersize=12, label='SELL SIGNAL')
plt.title('AIR LIQUIDE BB STRATEGY TRADING SIGNALS', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.show()

position = []
for i in range(len(bb_signal)):
    if bb_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)

for i in range(len(tsla['Close'])):
    if bb_signal[i] == 1:
        position[i] = 1
    elif bb_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i - 1]

upper_bb = tsla['upper_bb']
lower_bb = tsla['lower_bb']
close_price = tsla['Close']
bb_signal = pd.DataFrame(bb_signal).rename(columns={0: 'bb_signal'}).set_index(tsla.index)
position = pd.DataFrame(position).rename(columns={0: 'bb_position'}).set_index(tsla.index)

frames = [close_price, upper_bb, lower_bb, bb_signal, position]
strategy = pd.concat(frames, join='inner', axis=1)
strategy = strategy.reset_index().drop('Date', axis=1)

print(strategy.tail(7))

tsla_ret = pd.DataFrame(np.diff(tsla['Close'])).rename(columns={0: 'returns'})
bb_strategy_ret = []

for i in range(len(tsla_ret)):
    try:
        returns = tsla_ret['returns'][i] * strategy['bb_position'][i]
        bb_strategy_ret.append(returns)
    except:
        pass

bb_strategy_ret_df = pd.DataFrame(bb_strategy_ret).rename(columns={0: 'bb_returns'})

investment_value = 100000
number_of_stocks = math.floor(investment_value / tsla['Close'][-1])
bb_investment_ret = []

for i in range(len(bb_strategy_ret_df['bb_returns'])):
    returns = number_of_stocks * bb_strategy_ret_df['bb_returns'][i]
    bb_investment_ret.append(returns)

bb_investment_ret_df = pd.DataFrame(bb_investment_ret).rename(columns={0: 'investment_returns'})
total_investment_ret = round(sum(bb_investment_ret_df['investment_returns']), 2)
profit_percentage = math.floor((total_investment_ret / investment_value) * 100)
print(cl('Profit gained from the BB strategy by investing $100k in TSLA : {}'.format(total_investment_ret),
         attrs=['bold']))
print(cl('Profit percentage of the BB strategy : {}%'.format(profit_percentage), attrs=['bold']))


def get_benchmark(investment_value, start_date, end_date):
    spy = get_historic_data('^STOXX50E', start_date, end_date)
    spy = spy[spy.index >= start_date]['Close']
    benchmark = pd.DataFrame(np.diff(spy)).rename(columns={0: 'benchmark_returns'})

    investment_value = investment_value
    number_of_stocks = math.floor(investment_value / tsla['Close'][-1])
    benchmark_investment_ret = []

    for i in range(len(benchmark['benchmark_returns'])):
        returns = number_of_stocks * benchmark['benchmark_returns'][i]
        benchmark_investment_ret.append(returns)

    benchmark_investment_ret_df = pd.DataFrame(benchmark_investment_ret).rename(columns={0: 'investment_returns'})
    return benchmark_investment_ret_df


benchmark = get_benchmark(100000, '2020-01-01', '2020-12-31')

investment_value = 100000
total_benchmark_investment_ret = round(sum(benchmark['investment_returns']), 2)
benchmark_profit_percentage = math.floor((total_benchmark_investment_ret / investment_value) * 100)
print(cl('Benchmark profit by investing $100k : {}'.format(total_benchmark_investment_ret), attrs=['bold']))
print(cl('Benchmark Profit percentage : {}%'.format(benchmark_profit_percentage), attrs=['bold']))
print(cl('BB Strategy profit is {}% higher than the Benchmark Profit'.format(
    profit_percentage - benchmark_profit_percentage), attrs=['bold']))