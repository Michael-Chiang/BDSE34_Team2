import os

import numpy as np
import pandas as pd

aa = pd.read_csv('SOX.csv', index_col='Date', parse_dates=True)
aa = aa[['Price']]
aa.rename(columns={'Price': 'SOX'}, inplace=True)
aa.to_csv('SOX_test.csv')
NASDAQ = 'NASDAQ'
Dow_Jones = 'Dow_Jones'
SP_500 = 'S&P_500'
SOX = 'SOX_test'
USD_Index = 'USD_Index'
VIX = 'VIX'
WTI = 'WTI'
BADI = 'BADI'

output_file = 'indicator_data'

def merge_files(file_a, file_b, output_file):
    # Read files into DataFrames
    df_a = pd.read_csv(f'{file_a}.csv', index_col='Date', parse_dates=True)
    df_a = df_a[['Price']]
    df_a.rename(columns={'Price': file_a}, inplace=True)
    df_b = pd.read_csv(f'{file_b}.csv', index_col='Date', parse_dates=True)

    # Handle non-unique index in file B (optional)
    if not df_b.index.is_unique:
        df_b = df_b.set_index(['Date', df_b.columns[1]], drop=False)

    # # Drop extra dates from file A
    # extra_dates = df_a.index[~df_a.index.isin(df_b.index)]
    # df_a.drop(extra_dates, inplace=True)

    # # Merge DataFrames
    # merged_df = df_b.join(df_a, how='left', on='Date')

    merged_df = df_b.merge(df_a, on="Date")

    merged_df = merged_df.fillna(method='bfill')


    # Save merged data
    merged_df.to_csv(f'{output_file}.csv')



merge_files(USD_Index, SOX, output_file)
merge_files(VIX, output_file, output_file)
merge_files(WTI, output_file, output_file)
merge_files(BADI, output_file, output_file)
merge_files(NASDAQ, output_file, output_file)
merge_files(Dow_Jones, output_file, output_file)
merge_files(SP_500, output_file, output_file)