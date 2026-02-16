import pandas as pd
import numpy as np
from scipy import stats

file_path = "data/VIX_Nifty_from_2018.xlsx"
df = pd.read_excel(file_path)


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
print("Basic Shape before log:", df.shape)


df['Log_Return'] = np.log(df['Nifty'] / df['Nifty'].shift(1))
df = df.dropna()

print("After return computation shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 21-day rolling standard deviation
df['RV_21'] = df['Log_Return'].rolling(window=21).std()
df['RV_21'] = df['RV_21'] * np.sqrt(252)
df['RV_21_Fwd'] = df['RV_21'].shift(-21)
df = df.dropna()

print("After RV computation shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
#
df['IV'] = df['VIX'] / 100
print(df[['VIX','IV','RV_21_Fwd']].head())

# # Volatility Risk Premium
df['VRP'] = df['IV'] - df['RV_21_Fwd']
print(df[['IV','RV_21_Fwd','VRP']].head())
#
# #  average VRP
print("\nMean VRP:", df['VRP'].mean())
print("Median VRP:", df['VRP'].median())
print("Std Dev VRP:", df['VRP'].std())
#
# # One-sample t-test
t_stat, p_value = stats.ttest_1samp(df['VRP'], 0)
#
print("T-statistic:", t_stat)
print("P-value:", p_value)
#
# # VRP to daily return proxy
df['Option_Sell_Return'] = df['VRP'] / 252
df['Option_Buy_Return'] = -df['VRP'] / 252
df['Option_Sell_Cum'] = (1 + df['Option_Sell_Return']).cumprod()
df['Option_Buy_Cum'] = (1 + df['Option_Buy_Return']).cumprod()
#
print("Last 5 rows:")
print(df[['Option_Sell_Cum','Option_Buy_Cum']].tail())
#

def performance_stats(returns, cumulative):
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    skew = returns.skew()

    # Max Drawdown
    roll_max = cumulative.cummax()
    drawdown = cumulative / roll_max - 1
    max_dd = drawdown.min()

    return ann_return, ann_vol, sharpe, skew, max_dd
#
#
short_stats = performance_stats(df['Option_Sell_Return'], df['Option_Sell_Cum'])
long_stats = performance_stats(df['Option_Buy_Return'], df['Option_Buy_Cum'])
#
print("Option SELL Stats:")
print("Annual Return:", short_stats[0])
print("Annual Vol:", short_stats[1])
print("Sharpe:", short_stats[2])
print("Skew:", short_stats[3])
print("Max Drawdown:", short_stats[4])

print("\nOption BUY Stats:")
print("Annual Return:", long_stats[0])
print("Annual Vol:", long_stats[1])
print("Sharpe:", long_stats[2])
print("Skew:", long_stats[3])
print("Max Drawdown:", long_stats[4])
