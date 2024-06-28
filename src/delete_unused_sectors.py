import glob
import os

import numpy as np
import pandas as pd

metadata = pd.read_csv("stock_data_cleaned/symbol_sector.csv")
data_path = "stock_data_model/"
# Define the sectors to keep
sectors_to_keep = ["Finance", "Technology"]
all_csv_path = glob.glob(os.path.join(data_path, "*.csv"))
for csv_path in all_csv_path:
    # 取出文件名（包含扩展名）
    file_name_with_extension = os.path.basename(csv_path)

    # 去掉扩展名，取出文件名
    stockID = os.path.splitext(file_name_with_extension)[0]

    if (
        stockID not in metadata["Symbol"].values
        or metadata.loc[metadata["Symbol"] == stockID, "Sector"].values[0]
        not in sectors_to_keep
    ):
        # stockID = row["Symbol"]
        # sector = row["Sector"]
        # print(f"stockID = {stockID}, sector = {sector}")
        if os.path.exists(data_path + stockID + ".csv"):
            os.remove(data_path + stockID + ".csv")
            print(f"remove {stockID} successfully")
