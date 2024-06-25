import pandas as pd
import glob
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_economic_index(file_path):
    return pd.read_csv(file_path)


def get_file_paths(sector):
    return glob.glob(os.path.join(f"{sector}/**", "*.csv"), recursive=True)


def merge_and_save_data(file_path, economic_index):
    try:
        data = pd.read_csv(file_path)
        merged_data = pd.merge(data, economic_index, how='left', on='Date')
        merged_data.to_csv(file_path, index=False)
        logging.info(f"Finished merging {file_path}.")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")


def main():
    economic_index = load_economic_index('economic_index/daily_index.csv')
    sectors = ['Finance', 'Health Care', 'Technology']

    for sector in sectors:
        file_paths = get_file_paths(sector)
        for file_path in file_paths:
            merge_and_save_data(file_path, economic_index)


if __name__ == "__main__":
    main()
