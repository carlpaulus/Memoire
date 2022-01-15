# https://towardsdatascience.com/fibonacci-retracements-in-python-470eb33b6362
import yfinance as yf
import pandas as pd

import matplotlib.pyplot as plt
name = "GLD"

df = yf.download(name, start="2016-08-01", end="2020-02-15")
print(df)

highest_swing = -1
lowest_swing = -1

for i in range(1,df.shape[0]-1):
    if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1] and (highest_swing == -1 or df['High'][i] > df['High'][highest_swing]):
        highest_swing = i

    if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1] and (lowest_swing == -1 or df['Low'][i] < df['Low'][lowest_swing]):
        lowest_swing = i

ratios = [0,0.236, 0.382, 0.5 , 0.618, 0.786,1]
colors = ["black", "r", "g", "b", "cyan", "magenta", "yellow"]
levels = []

max_level = df['High'][highest_swing]
min_level = df['Low'][lowest_swing]

for ratio in ratios:
    if highest_swing > lowest_swing:  # Uptrend
        levels.append(max_level - (max_level-min_level)*ratio)
    else:  # Downtrend
        levels.append(min_level + (max_level-min_level)*ratio)

plt.rcParams['figure.figsize'] = [12, 7]

plt.rc('font', size=14)

plt.plot(df['Close'])
print(df)
start_date = df.index[min(highest_swing, lowest_swing)]
print(start_date)
end_date = df.index[max(highest_swing, lowest_swing)]
print(end_date)

print(levels)

for i in range(len(levels)):
    plt.hlines(levels[i], end_date, start_date, label="{:.1f}%".format(ratios[i]*100), colors=colors[i], linestyles="dashed")


plt.legend()
plt.show()

# [194.4499969482422, 173.81179766845702, 161.04409811401368, 150.7249984741211, 140.4058988342285, 125.71429934692382, 107.0]
