import numpy as np
import pandas as pd
import os

metadata = pd.read_csv("stock_data_cleaned/symbol_sector.csv")
data_path = 'stock_data_model/'
# Define the sectors to keep
sectors_to_keep = ['Finance', 'Health Care', 'Technology']

for index, row in metadata.iterrows():
    stockID = row['Symbol']
    sector = row['Sector']
    if os.path.exists(data_path + stockID + '.csv') and sector not in sectors_to_keep:
        os.remove(data_path + stockID + '.csv')
        print(f'remove {stockID} successfully')
