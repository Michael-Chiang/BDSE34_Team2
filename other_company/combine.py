# import os

# import numpy as np
# import pandas as pd

# with open("other_company.txt", "r") as f:
#     company = [line.strip() for line in f]


# def merge_files(file_a, file_b, output_file):
#     # Read files into DataFrames
#     df_a = pd.read_csv(file_a, index_col='Date', parse_dates=True)
#     df_a.drop(columns=['Open', 'High','Low', 'Adj Close', 'Volume'], inplace=True)
#     df_a.rename(columns={'Close': company}, inplace=True)

#     df_b = pd.read_csv(file_b, index_col='Date', parse_dates=True)

#     # Handle non-unique index in file B (optional)
#     if not df_b.index.is_unique:
#         df_b = df_b.set_index(['Date', df_b.columns[1]], drop=False)

#     # Drop extra dates from file A
#     extra_dates = df_a.index[~df_a.index.isin(df_b.index)]
#     df_a.drop(extra_dates, inplace=True)
    
#     print(df_a)
#     print(df_a.dtypes)


#     # Merge DataFrames
#     merged_df = df_b.join(df_a, how='left', on='Date')

#     merged_df = merged_df.fillna(method='bfill')

#     # Save merged data
#     merged_df.to_csv(output_file)


# # Replace 'file_a.csv' and 'file_b.csv' with your actual file paths
# file_a = f'{company}.csv'
# file_b = 'other_company.csv'
# output_file = 'other_company.csv'

# merge_files(file_a, file_b, output_file)



import os

import numpy as np
import pandas as pd


def merge_files(file_a, file_b, output_file):
    """
    Merges two CSV files based on the 'Date' column.

    Args:
        file_a (str): Path to the first CSV file.
        file_b (str): Path to the second CSV file (assumed to be 'other_company.csv').
        output_file (str): Path to the output CSV file.
    """

    # Read files into DataFrames
    df_a = pd.read_csv(file_a, index_col='Date', parse_dates=True)
    df_a.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
    df_a.rename(columns={'Close': company}, inplace=True)

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


with open("others_company.txt", "r") as f:
    companies = [line.strip() for line in f]  # List of companies

# Loop through each company
for company in companies:
    # Create file path for the company's data
    file_a = f'{company}.csv'

    # Ensure 'other_company.csv' exists (optional)
    if not os.path.exists(file_a):
        print(f"File not found: {file_a}. Skipping...")
        continue

    # Merge files and update output file
    merge_files(file_a, 'other_company.csv', 'other_company.csv')
    print(f"Merged data for {company}")

print("Process completed!")


