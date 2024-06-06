#import package
import yfinance as yf
import os
import csv

#path of stock tickers 
path = 'stock_tickers.txt'

#path of data
data_dir = 'stock_data'

# File to store ticker and df_len
output_file = 'ticker_lengths.csv'

#make a directory
os.makedirs(data_dir, exist_ok=True)


#download data
with open(path, 'r') as f, open(output_file, 'w', newline='') as csvfile:
    lines = f.readlines()
    writer = csv.writer(csvfile)
    writer.writerow(['Ticker', 'Data Length'])  # Write header
   for line in lines:
        ticker = line.strip("\n")
        print(f'Processing {ticker}...')
        try:
            df = yf.download(ticker, start='1984-06-01', end='2024-05-31')
            df_len = len(df)
            if df.empty:
                print(f'{ticker} has no data available.')
                writer.writerow([ticker, 'No data available'])
                continue
            df.to_csv(os.path.join(data_dir, ticker + '.csv'))
            print(f'Data for {ticker} saved successfully.')
            print(df_len)
            writer.writerow([ticker, df_len])
        except Exception as e:
            print(f'Error processing {ticker}: {e}')
            writer.writerow([ticker, f'Error: {e}'])
            continue
