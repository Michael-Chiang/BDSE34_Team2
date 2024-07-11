import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

tech = {0:'DOCN', 1:'AGYS', 2:'RPAY', 3:'EVER', 4:'CACI'}
fin =  {0:'NVG', 1:'GLAD', 2:'TY', 3:'CGBD', 4:'MCAA'}
QQQ = pd.read_csv('C:/Share/期末專題/index_3/QQQ.csv', index_col='Date', parse_dates=['Date'])
QQQ.sort_index(inplace = True)
XLF = pd.read_csv('C:/Share/期末專題/index_3/XLF.csv', index_col='Date', parse_dates=['Date'])
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
        plt.plot(df.index, df['Adj Close'], label=f'Technology_cluster_{clusterID}')   
plt.plot(QQQ.index, QQQ['Price'], label='QQQ', color='black', linewidth=2)
plt.title("Technology hurst's 5 cluster & QQQ")
plt.xlabel('Date')
plt.ylabel('Standardized Adj Close')
plt.legend()   
plt.savefig("pic/tech/Technology hurst's 5 cluster & QQQ_new.png", format='png')
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
        plt.plot(df.index, df['Adj Close'], label=f'Finance_cluster_{clusterID}')   
plt.plot(XLF.index, XLF['Price'], label='XLF', color='black', linewidth=2)
plt.title("TFinance hurst's 5 cluster & XLF")
plt.xlabel('Date')
plt.ylabel('Standardized Adj Close')
plt.legend()   
plt.savefig("pic/fin/Finance hurst's 5 cluster & XLF_new.png", format='png')
plt.close()