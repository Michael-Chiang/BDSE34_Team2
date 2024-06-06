import pandas as pd
from glob import glob
 
files = glob('stock*.csv')
 
df = pd.concat((pd.read_csv(file, dtype={'Ticker': str}) for file in files), axis='columns')
df.to_csv('stock_info_all.csv')