import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def create_directory(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory {path} created or already exists.")
    except OSError as e:
        logging.error(f"Error creating directory {path}: {e}")
        sys.exit(1)


def read_csv(file_path):
    if not file_path.exists():
        logging.error(f"File {file_path} not found.")
        return None
    return pd.read_csv(file_path)


def process_stock_data(stock_data, metadata, fold_num, split_value):
    stock_data_filtered = stock_data[stock_data[f'split_{fold_num}']
                                     == split_value].iloc[:, :-5]
    replicated_metadata = pd.concat(
        [metadata] * len(stock_data_filtered), ignore_index=True)
    combined_df = pd.concat([replicated_metadata.reset_index(
        drop=True), stock_data_filtered.reset_index(drop=True)], axis=1)
    return combined_df


def main(fold_num):

    metadata_path = Path("stock_data_cleaned/symbol_sector.csv")
    metadata = read_csv(metadata_path)
    if metadata is None:
        sys.exit(1)

    data_path = Path('stock_data_model_split/')
    sectors = ['Finance', 'Health Care', 'Technology']

    for sector in sectors:
        parent_path = Path(sector)
        create_directory(parent_path)

        split_path = parent_path / f'split_{fold_num}/'
        create_directory(split_path)

        whole_dfs = {'train': pd.DataFrame(), 'valid': pd.DataFrame()}

        for _, row in metadata.iterrows():
            stockID = row['Symbol']
            if row['Sector'] != sector:
                continue

            stock_file_path = data_path / f'{stockID}.csv'
            stock_data = read_csv(stock_file_path)
            if stock_data is None:
                continue

            stock_metadata = metadata[metadata['Symbol'] == stockID]

            for split_name, split_value in zip(['train', 'valid'], [1, 2]):
                combined_df = process_stock_data(
                    stock_data, stock_metadata, fold_num, split_value)
                whole_dfs[split_name] = pd.concat(
                    [whole_dfs[split_name], combined_df], axis=0, ignore_index=True)

            logging.info(
                f"Processed stockID = {stockID} for fold_num = {fold_num}")

        logging.info("Processing finished.")

        try:
            for split_name in ['train', 'valid']:
                output_file = split_path / \
                    f'stock_all_{split_name}_{fold_num}.csv'
                whole_dfs[split_name].to_csv(output_file, index=False)
            logging.info(f"Files saved to {split_path}")
        except Exception as e:
            logging.error(f"Error saving files: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logging.error('No fold number input.')
        sys.exit(1)

    fold_num = sys.argv[1]
    main(fold_num)
