# https://medium.com/codex/algorithmic-trading-with-relative-strength-index-in-python-d969cf22dd85

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from math import floor
from termcolor import colored as cl

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)


def get_historical_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)


ibm = get_historical_data('SAN.PA', '2020-01-01', '2021-12-31')
# print(ibm)

# RSI calculation
def get_rsi(close, lookback):
    ret = close.diff()
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
    rsi_df = pd.DataFrame(rsi).rename(columns={0: 'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()
    return rsi_df[3:]


ibm['rsi_14'] = get_rsi(ibm['Close'], 14)
ibm = ibm.dropna()
# print(ibm)

# RSI plot
ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=4, colspan=1)
ax1.plot(ibm['Close'], linewidth=2.5)
ax1.set_title('IBM CLOSE PRICE')
ax2.plot(ibm['rsi_14'], color='orange', linewidth=2.5)
ax2.axhline(30, linestyle='--', linewidth=1.5, color='grey')
ax2.axhline(70, linestyle='--', linewidth=1.5, color='grey')
ax2.set_title('IBM RELATIVE STRENGTH INDEX')
# plt.show()


# Creating the trading strategy
def implement_rsi_strategy(prices, rsi):
    buy_price = []
    sell_price = []
    rsi_signal = []
    signal = 0

    for i in range(len(rsi)):
        if rsi[i - 1] > 30 and rsi[i] < 30:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
        elif rsi[i - 1] < 70 and rsi[i] > 70:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            rsi_signal.append(0)

    return buy_price, sell_price, rsi_signal


buy_price, sell_price, rsi_signal = implement_rsi_strategy(ibm['Close'], ibm['rsi_14'])

#Plotting the trading signals
ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=4, colspan=1)
ax1.plot(ibm['Close'], linewidth=2.5, color='skyblue', label='SAN.PA')
ax1.plot(ibm.index, buy_price, marker='^', markersize=12, linewidth=0, color='green', label='BUY SIGNAL')
ax1.plot(ibm.index, sell_price, marker='v', markersize=12, linewidth=0, color='r', label='SELL SIGNAL')
ax1.set_title('SANOFI TRADE SIGNALS')
ax1.legend(loc='lower left', fontsize=12)
ax2.plot(ibm['rsi_14'], color='orange', linewidth=2.5, label='RSI')
ax2.axhline(30, linestyle='--', linewidth=1.5, color='grey')
ax2.axhline(70, linestyle='--', linewidth=1.5, color='grey')
ax2.legend(loc='lower right', fontsize=12)
ax2.set_title('Sanofi RSI')
plt.show()

# Creating our position
position = []
for i in range(len(rsi_signal)):
    if rsi_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)

for i in range(len(ibm['Close'])):
    if rsi_signal[i] == 1:
        position[i] = 1
    elif rsi_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i - 1]

rsi = ibm['rsi_14']
close_price = ibm['Close']
rsi_signal = pd.DataFrame(rsi_signal).rename(columns={0: 'rsi_signal'}).set_index(ibm.index)
position = pd.DataFrame(position).rename(columns={0: 'rsi_position'}).set_index(ibm.index)

frames = [close_price, rsi, rsi_signal, position]
strategy = pd.concat(frames, join='inner', axis=1)
# print(strategy.head())

# Backtesting
ibm_ret = pd.DataFrame(np.diff(ibm['Close'])).rename(columns={0: 'returns'})
rsi_strategy_ret = []

for i in range(len(ibm_ret)):
    returns = ibm_ret['returns'][i] * strategy['rsi_position'][i]
    rsi_strategy_ret.append(returns)

rsi_strategy_ret_df = pd.DataFrame(rsi_strategy_ret).rename(columns={0: 'rsi_returns'})
investment_value = 100000
number_of_stocks = floor(investment_value / ibm['Close'][-1])
rsi_investment_ret = []

for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
    returns = number_of_stocks * rsi_strategy_ret_df['rsi_returns'][i]
    rsi_investment_ret.append(returns)

rsi_investment_ret_df = pd.DataFrame(rsi_investment_ret).rename(columns={0: 'investment_returns'})
total_investment_ret = round(sum(rsi_investment_ret_df['investment_returns']), 2)
profit_percentage = floor((total_investment_ret / investment_value) * 100)
print(cl('Profit gained from the RSI strategy by investing $100k in IBM : {}'.format(total_investment_ret),
         attrs=['bold']))
print(cl('Profit percentage of the RSI strategy : {}%'.format(profit_percentage), attrs=['bold']))