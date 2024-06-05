import yfinance as yf
import os

path='6000.txt'
data_dir = './test'

with open(path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ticker = line.strip("\n")
        print(ticker)
        try:
            df = yf.download(ticker, start='2005-10-01', end='2024-05-31')
            df.to_csv(os.path.join(data_dir, ticker+'.csv'))
            print(df)
        except:
            print(f'{ticker} not found')
            continue