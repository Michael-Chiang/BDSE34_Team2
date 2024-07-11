import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

tech = {'0~0.33':'PANW', '0.33~0.66':'AAPL', '0.66~1':'NVDA'}
fin =  {'0~0.33':'BX', '0.33~0.66':'JPM', '0.66~1':'WFC'}
QQQ = pd.read_csv('C:/Share/期末專題/daily_data_combine/QQQ.csv', index_col='Date', parse_dates=['Date'])
QQQ.sort_index(inplace = True)
XLF = pd.read_csv('C:/Share/期末專題/daily_data_combine/XLF.csv', index_col='Date', parse_dates=['Date'])
XLF.sort_index(inplace = True)

start_date = '2024-01-02'
end_date = '2024-06-20'

# QQQ & XLF 標準化
QQQ = QQQ.loc[start_date:end_date]
XLF = XLF.loc[start_date:end_date]

scaler = StandardScaler()
QQQ['Price'] = scaler.fit_transform(QQQ[['Price']])
XLF['Price'] = scaler.fit_transform(XLF[['Price']])

# 科技股 hurst 分群標化後與 QQQ 的趨勢圖
plt.figure(figsize=(14, 7))
for clusterID, symbol in tech.items():
    filepath = os.path.join('C:/Share/期末專題/tech_index_signal/stock_data_model_new/Technology_new', f"{symbol}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col='Date', parse_dates=['Date'])
        df = df.loc[start_date:end_date]
        # 标准化 Adj Close
        scaler = StandardScaler()
        df['Adj Close'] = scaler.fit_transform(df[['Adj Close']])
        plt.plot(df.index, df['Adj Close'], label=f'Technology hurst exponent {clusterID}')   
plt.plot(QQQ.index, QQQ['Price'], label='QQQ', color='black', linewidth=2)
plt.title("Technology hurst's 3 cluster & QQQ")
plt.xlabel('Date')
plt.ylabel('Standardized Adj Close')
plt.legend()   
plt.savefig("pic/tech/Technology hurst's 3 cluster & QQQ.png", format='png')
plt.close()

# 金融股 hurst 分群標化後與 XLF 的趨勢圖
plt.figure(figsize=(14, 7))
for clusterID, symbol in fin.items():
    filepath = os.path.join('C:/Share/期末專題/tech_index_signal/stock_data_model_new/Finance_new', f"{symbol}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col='Date', parse_dates=['Date'])
        df = df.loc[start_date:end_date]
        # 标准化 Adj Close
        scaler = StandardScaler()
        df['Adj Close'] = scaler.fit_transform(df[['Adj Close']])
        plt.plot(df.index, df['Adj Close'], label=f'Finance hurst exponent {clusterID}')   
plt.plot(XLF.index, XLF['Price'], label='XLF', color='black', linewidth=2)
plt.title("TFinance hurst's 3 cluster & XLF")
plt.xlabel('Date')
plt.ylabel('Standardized Adj Close')
plt.legend()   
plt.savefig("pic/fin/Finance hurst's 3 cluster & XLF.png", format='png')
plt.close()