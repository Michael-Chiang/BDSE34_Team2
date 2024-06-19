import datetime as dt

import numpy as np
import pandas as pd
import requests
from fredapi import Fred

# 填入專屬 API，讓 fredapi 核准會員通過
api_key = '1ffb47ea0aba23fc24d448a85cf60324'
fred = Fred(api_key)


FEDFUNDS = fred.get_series('FEDFUNDS')
PCE = fred.get_series('PCE')
UNRATE = fred.get_series('UNRATE')
CPIAUCSL = fred.get_series('CPIAUCSL')
PAYEMS= fred.get_series('PAYEMS')

data_dict = {
  'FEDFUNDS': FEDFUNDS,
  'PCE': PCE,
  'UNRATE': UNRATE,
  'CPIAUCSL': CPIAUCSL,
  'PAYEMS': PAYEMS,
}


# Create a DataFrame
data = pd.DataFrame.from_dict(data_dict)

data.index.name = "Date"

# Filter data for the specified time range
data = data.loc[(data.index >= '2004-01-01') & (data.index <= '2024-05-30')]

# Get the first day of each month
month_starts = data.index.to_period('M').to_timestamp()
month_starts = month_starts.drop_duplicates()
print(month_starts)


# Select the first data point for each month
first_of_month_data = data.loc[month_starts]

# Print the first data point of each month
print(first_of_month_data)
# Save the resampled data to CSV
first_of_month_data.to_csv("FRED.csv")
