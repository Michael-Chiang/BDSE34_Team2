import numpy as np
import pandas as pd
import os

metadata = pd.read_csv("stock_data_cleaned/stock_vector_and_encoding.csv")

dummy_cols = ['Basic Materials', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Finance',  'Health Care',	'Industrials', 'Miscellaneous', 'Real Estate', 'Technology', 'Telecommunications', 'Utilities'
              ]
data = pd.from_dummies(metadata[dummy_cols], default_category='Sector')
data.columns = ['Sector']
data = pd.concat([metadata['Symbol'], data], axis=1)

data.to_csv('stock_data_cleaned/symbol_sector.csv', index=False)
