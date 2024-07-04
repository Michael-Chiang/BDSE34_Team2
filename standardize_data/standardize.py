import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取分群对照表
tech_clusterID_df = pd.read_csv('stock_all_info_technology.csv')
finance_clusterID_df = pd.read_csv('stock_all_info_finance.csv')

# 创建一个空的字典来存储每个分群的DataFrame
tech_clusterID = {i: pd.DataFrame() for i in range(-1,5)}
finance_clusterID = {i: pd.DataFrame() for i in range(-1,5)}

# 设定时间范围
start_date = '2020-01-02'
end_date = '2024-06-20'

# 读取科技股数据并进行处理
tech_folder = 'C:/Share/期末專題/tech_index_signal/stock_data_model_new/Technology_new'  # 替换为实际的文件夹路径
for _, row in tech_clusterID_df.iterrows():
    symbol = row['Symbol']
    clusterID = row['clusterID']  # 分组信息
    filepath = os.path.join(tech_folder, f"{symbol}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col='Date', parse_dates=['Date'])
        df = df.loc[start_date:end_date]
        # 标准化 Adj Close
        scaler = StandardScaler()
        df['Adj Close'] = scaler.fit_transform(df[['Adj Close']])
        tech_clusterID[clusterID][symbol] = df['Adj Close']

# 读取财金股数据并进行处理
finance_folder = 'C:/Share/期末專題/tech_index_signal/stock_data_model_new/Finance_new'  # 替换为实际的文件夹路径
for _, row in finance_clusterID_df.iterrows():
    symbol = row['Symbol']
    clusterID = row['clusterID']  # 分组信息
    filepath = os.path.join(finance_folder, f"{symbol}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col='Date', parse_dates=['Date'])
        df = df.loc[start_date:end_date]
        # 标准化 Adj Close
        scaler = StandardScaler()
        df['Adj Close'] = scaler.fit_transform(df[['Adj Close']])
        finance_clusterID[clusterID][symbol] = df['Adj Close']

# 计算分群的加总平均值并绘制趋势图
clusterID_to_plot = [0, 1, 2, 3, 4]  # 指定要绘制的group
for clusterID, data in tech_clusterID.items():
    if clusterID in clusterID_to_plot and not data.empty:
        data['Average'] = data.mean(axis=1)
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['Average'], label=f'Technology clusterID {clusterID}')
        plt.title(f'Technology clusterID {clusterID} Average Adj Close')
        plt.xlabel('Date')
        plt.ylabel('Standardized Adj Close')
        plt.legend()
        plt.show()

clusterID_to_plot = [0, 1, 2, 3, 4]  # 指定要绘制的group
for clusterID, data in finance_clusterID.items():
    if clusterID in clusterID_to_plot and not data.empty:
        data['Average'] = data.mean(axis=1)
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['Average'], label=f'Finance clusterID {clusterID}')
        plt.title(f'Finance clusterID {clusterID} Average Adj Close')
        plt.xlabel('Date')
        plt.ylabel('Standardized Adj Close')
        plt.legend()
        plt.show()

# 计算分群的加总平均值并绘制趋势图
plt.figure(figsize=(14, 7))
clusterID_to_plot = [0, 1, 2, 3, 4]  # 指定要绘制的group
for clusterID, data in tech_clusterID.items():
    if clusterID in clusterID_to_plot and not data.empty:
        data['Average'] = data.mean(axis=1)
        plt.plot(data.index, data['Average'], label=f'Technology clusterID {clusterID}')
plt.title(f'Technology clusterID {clusterID} Average Adj Close')
plt.xlabel('Date')
plt.ylabel('Standardized Adj Close')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
clusterID_to_plot = [0, 1, 2, 3, 4]  # 指定要绘制的group
for clusterID, data in finance_clusterID.items():
    if clusterID in clusterID_to_plot and not data.empty:
        data['Average'] = data.mean(axis=1)
        plt.plot(data.index, data['Average'], label=f'Finance clusterID {clusterID}')
plt.title(f'Finance clusterID {clusterID} Average Adj Close')
plt.xlabel('Date')
plt.ylabel('Standardized Adj Close')
plt.legend()
plt.show()
