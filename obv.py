# OBV: On-balance Volume - https: // www.youtube.com / watch?v = MRGXd8eaWB4

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


def get_historical_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)


df = get_historical_data('SU.PA', '2020-04-01', '2020-09-30')
# print(df)

# Calculate OBV: Buy whenever the OBV is above the OBV_EMA
obv = [0]

# loop through the data set (close price) from the second row (index 1) to the end of the set
for i in range(1, len(df['Close'])):
    if df['Close'][i] > df['Close'][i - 1]:
        obv.append(obv[-1] + df['Volume'][i])
    elif df['Close'][i] < df['Close'][i - 1]:
        obv.append(obv[-1] - df['Volume'][i])
    else:
        obv.append(obv[-1])

# Store the obv and OBV_EMA into new columns
df['OBV'] = obv
df['OBV_EMA'] = df['OBV'].ewm(span=50).mean()


# Create a function to signal when to buy (OBV > OBV_EMA) or sell (OBV < OBV_EMA) the stock (Else Do Nothing)
def buy_sell(signal, col1, col2):
    sig_price_buy = []
    sig_price_sell = []
    flag = -1

    # Loop through the length of the data set
    for i in range(0, len(signal)):
        # If OBV > OBV_EMA Then Buy --> col1 => 'OBV' and col2 => 'OBV_EMA'
        if signal[col1][i] > signal[col2][i] and flag != 1:
            sig_price_buy.append(signal['Close'][i])
            sig_price_sell.append(np.nan)
            flag = 1
        # If OBV < OBV_EMA Then Sell
        elif signal[col1][i] < signal[col2][i] and flag != 0:
            sig_price_sell.append(signal['Close'][i])
            sig_price_buy.append(np.nan)
            flag = 0
        else:
            sig_price_sell.append(np.nan)
            sig_price_buy.append(np.nan)

    return sig_price_buy, sig_price_sell


# Create buy and sell columns
x = buy_sell(df, 'OBV', 'OBV_EMA')

df['Buy_Signal_Price'] = x[0]
df['Sell_Signal_Price'] = x[1]

# Create and plot buy and sell price
ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)

ax1.plot(df['Close'], color='blue', linewidth=2, label='SU.PA')
#ax1.plot(df["Close"], label='Close', alpha=0.35, color='blue')
ax1.plot(df.index, df['Buy_Signal_Price'], label="Buy Signal", marker='^', markersize=12, linewidth=0, color="green")
ax1.plot(df.index, df['Sell_Signal_Price'], label="Sell Signal", marker='v', markersize=12, linewidth=0, color='red')
ax1.set_title('SCHNEIDER ELECTRIC TRADING SIGNALS')
ax1.legend(loc='upper left', fontsize=12)

ax2.plot(df["OBV"], label='OBV', color='orange', linewidth=2)
ax2.plot(df["OBV_EMA"], label='OBV_EMA', color='purple', linewidth=2)
ax2.set_title('ON-BALANCE VOLUME INDICATOR')
ax2.legend(loc='lower right', fontsize=12)

plt.show()
