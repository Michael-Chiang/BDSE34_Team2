import os

import numpy as np
import pandas as pd


def merge_files(file_a, file_b, output_file):
    # Read files into DataFrames
    df_a = pd.read_csv(file_a, index_col='Date', parse_dates=True)
    df_a.rename(columns={'Price': 'S&P_500'}, inplace=True)
    df_b = pd.read_csv(file_b, index_col='Date', parse_dates=True)

    # Handle non-unique index in file B (optional)
    if not df_b.index.is_unique:
        df_b = df_b.set_index(['Date', df_b.columns[1]], drop=False)

    # Drop extra dates from file A
    extra_dates = df_a.index[~df_a.index.isin(df_b.index)]
    df_a.drop(extra_dates, inplace=True)

    # Merge DataFrames
    merged_df = df_b.join(df_a, how='left', on='Date')

    merged_df = merged_df.fillna(method='bfill')

    # Save merged data
    merged_df.to_csv(output_file)


# Replace 'file_a.csv' and 'file_b.csv' with your actual file paths
file_a = 'daily_index.csv'
file_b = 'index_3.csv'
output_file = 'indicator_data.csv'

merge_files(file_a, file_b, output_file)
