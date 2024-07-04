import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 讀取指數數據
indicator_df = pd.read_csv('indicator_data.csv', index_col=['Date'], parse_dates=['Date'])
indicator_df = indicator_df.sort_index()

folder_tech = 'C:/Share/期末專題/tech_index_signal/stock_data_model_new/Technology_new'
correlation_all_Adj_Close = pd.DataFrame()
correlation_tech_Adj_Close = pd.DataFrame()
for filename in os.listdir(folder_tech):
    if filename.endswith('.csv'):
        filepath = os.path.join(folder_tech, filename)
        df = pd.read_csv(filepath, index_col=['Date'], parse_dates=['Date'])
        Symbol = os.path.splitext(filename)[0]
        df = df[['Adj Close']]
        # 確保時間索引是排序的
        df = df.sort_index()

        # 設置時間範圍
        start_date = '2020-01-02'
        end_date = '2024-06-20'

        # 檢查時間範圍是否在時間索引中
        if start_date in df.index and end_date in df.index and start_date in indicator_df.index and end_date in indicator_df.index:
        # 篩選時間範圍內的數據
            df = df.loc[start_date:end_date]
            indicator_df = indicator_df.loc[start_date:end_date]
        else:
            print(f"指定的日期範圍不在{Symbol}時間索引中")

        # 合併數據，假設使用日期進行合併
        merged_df = pd.merge(df, indicator_df, on='Date')
        print(merged_df)
        # 計算個股調整後收盤價和指標之間的相關性
        correlation_matrix = merged_df.corr()
        correlation_Adj_Close = correlation_matrix['Adj Close'].to_frame().transpose()
        correlation_Adj_Close['Symbol'] = Symbol
        correlation_Adj_Close['type'] = 'tech'
        # 取消索引設定
        correlation_Adj_Close.reset_index(drop=True, inplace=True)
        
        correlation_tech_Adj_Close = pd.concat([correlation_tech_Adj_Close, correlation_Adj_Close])
        correlation_tech_Adj_Close.reset_index(drop=True, inplace=True)

        correlation_all_Adj_Close = pd.concat([correlation_all_Adj_Close, correlation_Adj_Close])
        correlation_all_Adj_Close.reset_index(drop=True, inplace=True)

    
print(correlation_tech_Adj_Close)
correlation_tech_Adj_Close.to_csv('Correlation_tech_Adj_Close.csv')

folder_fin = 'C:/Share/期末專題/tech_index_signal/stock_data_model_new/Finance_new'
correlation_fin_Adj_Close = pd.DataFrame()
for filename in os.listdir(folder_fin):
    if filename.endswith('.csv'):
        filepath = os.path.join(folder_fin, filename)
        df = pd.read_csv(filepath, index_col=['Date'], parse_dates=['Date'])
        Symbol = os.path.splitext(filename)[0]
        df = df[['Adj Close']]
        # 確保時間索引是排序的
        df = df.sort_index()

        # 設置時間範圍
        start_date = '2020-01-02'
        end_date = '2024-06-20'

        # 檢查時間範圍是否在時間索引中
        if start_date in df.index and end_date in df.index and start_date in indicator_df.index and end_date in indicator_df.index:
        # 篩選時間範圍內的數據
            df = df.loc[start_date:end_date]
            indicator_df = indicator_df.loc[start_date:end_date]
        else:
            print(f"指定的日期範圍不在{Symbol}時間索引中")

        # 合併數據，假設使用日期進行合併
        merged_df = pd.merge(df, indicator_df, on='Date')
        print(merged_df)
        # 計算個股調整後收盤價和指標之間的相關性
        correlation_matrix = merged_df.corr()
        correlation_Adj_Close = correlation_matrix['Adj Close'].to_frame().transpose()
        correlation_Adj_Close['Symbol'] = Symbol
        correlation_Adj_Close['type'] = 'fin'
        # 取消索引設定
        correlation_Adj_Close.reset_index(drop=True, inplace=True)
        
        correlation_fin_Adj_Close = pd.concat([correlation_fin_Adj_Close, correlation_Adj_Close])
        correlation_fin_Adj_Close.reset_index(drop=True, inplace=True)

        correlation_all_Adj_Close = pd.concat([correlation_all_Adj_Close, correlation_Adj_Close])
        correlation_all_Adj_Close.reset_index(drop=True, inplace=True)
    

    
print(correlation_fin_Adj_Close)
correlation_fin_Adj_Close.to_csv('Correlation_fin_Adj_Close.csv')

print(correlation_all_Adj_Close)
correlation_all_Adj_Close.to_csv('Correlation_all_Adj_Close.csv')


