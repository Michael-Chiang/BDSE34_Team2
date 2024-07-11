import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

QQQ = pd.read_csv('C:/Share/期末專題/daily_data_combine/QQQ.csv', index_col='Date', parse_dates=True)
QQQ = QQQ[['Price']]
XLF = pd.read_csv('C:/Share/期末專題/daily_data_combine/XLF.csv', index_col='Date', parse_dates=True)
XLF = XLF[['Price']]
daily = pd.read_csv('indicator_data.csv', index_col='Date', parse_dates=True)

QQQ = QQQ.merge(daily, on='Date')
XLF = XLF.merge(daily, on='Date')

QQQ = QQQ.sort_index()
XLF = XLF.sort_index()

start_date = '2014-01-02'
end_date = '2024-06-28'

QQQ = QQQ.loc[start_date:end_date]
XLF = XLF.loc[start_date:end_date]

correlation_matrix_QQQ = QQQ.corr()
correlation_matrix_XLF = XLF.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_QQQ, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("QQQ's correaltion_coefficient_with_daily_index")
plt.savefig("QQQ's correaltion_coefficient_with_daily_index.png", format = 'png')


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_XLF, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("XLF's correaltion_coefficient_with_daily_index")
plt.savefig("XLF's correaltion_coefficient_with_daily_index.png", format = 'png')