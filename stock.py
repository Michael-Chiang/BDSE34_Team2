import yfinance as yf
import os
path = '6000.txt'
data_dir = '500'
os.makedirs(data_dir, exist_ok=True)

with open(path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ticker = line.strip("\n")
        print(f'Processing {ticker}...')
        try:
            df = yf.download(ticker, start='1985-06-01', end='2024-05-31')
            if df.empty:
                print(f'{ticker} has no data available.')
                continue
            df.to_csv(os.path.join(data_dir, ticker + '.csv'))
            print(f'Data for {ticker} saved successfully.')
        except Exception as e:
            print(f'Error processing {ticker}: {e}')
            continue
