#import package
import yfinance as yf
import os
#path of stock tickers 
path = '6000.txt'
#path of data
data_dir = '500'
#make a directory
os.makedirs(data_dir, exist_ok=True)

#download data
with open(path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ticker = line.strip("\n")
        print(f'Processing {ticker}...')
        try:
            df = yf.download(ticker, start='1985-06-01', end='2024-05-31')
            if df.empty:
                print(f'{ticker} has no data available.')#this stock ticker doesn't have any data
                continue
            df.to_csv(os.path.join(data_dir, ticker + '.csv'))
            print(f'Data for {ticker} saved successfully.')
        except Exception as e:
            print(f'Error processing {ticker}: {e}')
            continue
