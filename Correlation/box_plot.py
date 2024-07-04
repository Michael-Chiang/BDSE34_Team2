import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

corr = pd.read_csv('Correlation_all_Adj_Close.csv')

print(f'tech：{len(corr)} 筆')
print(corr)


sns.boxplot(x='type', y='NASDAQ', data=corr)
corr.NASDAQ.describe()

sns.boxplot(x='type', y='Dow_Jones', data=corr)
corr.Dow_Jones.describe()

corr_sp500 = corr['S&P_500']
sns.boxplot(x='type', y='S&P_500', data=corr)
corr_sp500.describe()

sns.boxplot(x='type', y='SOX', data=corr)
corr.SOX.describe()

sns.boxplot(x='type', y='USD_index', data=corr)
corr.USD_index.describe()

sns.boxplot(x='type', y='VIX', data=corr)
corr.VIX.describe()

sns.boxplot(x='type', y='BADI', data=corr)
corr.BADI.describe()