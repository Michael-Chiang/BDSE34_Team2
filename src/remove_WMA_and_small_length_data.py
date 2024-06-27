import glob
import os
import sys

import pandas as pd


def process_files(directory, length_thr):
    file_paths = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)

    for file_path in file_paths:
        try:
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Drop WMA columns and the 'Unnamed: 0' column if they exist
            WMA_cols = [col for col in data.columns if "WMA" in col]
            columns_to_drop = WMA_cols + ["Unnamed: 0"]
            data.drop(
                columns=[col for col in columns_to_drop if col in data.columns],
                inplace=True,
            )

            # Filter symbols based on the threshold
            symbol_counts = data["Symbol"].value_counts()
            symbols_to_keep = symbol_counts[symbol_counts >= length_thr].index
            data_cleaned = data[data["Symbol"].isin(symbols_to_keep)]

            # Save the cleaned data back to the file
            data_cleaned.to_csv(file_path, index=False)
            print(f"Finished processing {file_path}.")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["Finance", "Technology"]:
        print("Incorrect input")
        sys.exit()

    # Set the length threshold based on the argument
    length_thr = 70 if sys.argv[1] == "Finance" else 25
    directory = sys.argv[1]

    # Process the files
    process_files(directory, length_thr)


if __name__ == "__main__":
    main()
