import csv
import os
import pandas as pd
import talib as ta
import yfinance as yf


def download_stock_data(ticker, data_dir, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return ticker, 0, "", "", ""

        # Calculate technical indicators

        # Simple Moving Averages
        df["Close_SMA5"] = ta.SMA(df["Close"], timeperiod=5)
        df["Close_SMA20"] = ta.SMA(df["Close"], timeperiod=20)
        df["Close_SMA60"] = ta.SMA(df["Close"], timeperiod=60)
        df["Close_SMA120"] = ta.SMA(df["Close"], timeperiod=120)
        df["Close_SMA180"] = ta.SMA(df["Close"], timeperiod=180)

        # Exponential Moving Averages
        df["Close_EMA5"] = ta.EMA(df["Close"], timeperiod=5)
        df["Close_EMA20"] = ta.EMA(df["Close"], timeperiod=20)
        df["Close_EMA60"] = ta.EMA(df["Close"], timeperiod=60)
        df["Close_EMA120"] = ta.EMA(df["Close"], timeperiod=120)
        df["Close_EMA180"] = ta.EMA(df["Close"], timeperiod=180)

        # Relative Strength Index
        df["RSI6"] = ta.RSI(df["Close"], timeperiod=6)
        df["RSI14"] = ta.RSI(df["Close"], timeperiod=14)
        df["RSI24"] = ta.RSI(df["Close"], timeperiod=24)

        # MACD
        df["MACD"], df["DIF"], df["OSC_Hist"] = ta.MACD(
            df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Bollinger Bands
        df["Upper_Band"], df["Middle_Band"], df["Lower_Band"] = ta.BBANDS(
            df["Close"], timeperiod=20
        )

        # Stochastic Oscillator
        df["Stochastic_K"], df["Stochastic_D"] = ta.STOCH(
            df["High"],
            df["Low"],
            df["Close"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )

        # Williams %R
        df["Williams_R"] = ta.WILLR(df["High"], df["Low"], df["Close"], timeperiod=14)

        # ATR
        df["ATR"] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)

        # MFI
        df["MFI"] = ta.MFI(
            df["High"], df["Low"], df["Close"], df["Volume"], timeperiod=14
        )

        # OBV
        df["OBV"] = ta.OBV(df["Close"], df["Volume"])

        # Momentum
        df["Momentum"] = ta.MOM(df["Close"], timeperiod=10)

        # ROC
        df["ROC"] = ta.ROC(df["Close"], timeperiod=10)

        # ADX
        df["ADX"] = ta.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)

        # DMI
        df["PLUS_DMI"] = ta.PLUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)
        df["MINUS_DMI"] = ta.MINUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)

        # Daily Return and Change
        df["Daily_Return"] = df["Close"].pct_change()
        df["Daily_Change"] = df["Close"].diff()

        # Calculate the rolling standard deviation of the closing prices
        df["Rolling_STD_Dev"] = df["Close"].rolling(window=20).std()

        # Drop rows with any NaN values
        # df.dropna(inplace=True)

        if df.empty:
            return ticker, 0, "", "", ""

        # Save data with indicators
        df.to_csv(f"{data_dir}/{ticker}.csv")

        first_date = df.index[0].strftime("%Y-%m-%d")
        last_date = df.index[-1].strftime("%Y-%m-%d")
        return ticker, len(df), first_date, last_date, df["Rolling_STD_Dev"].iloc[-1]
    except Exception as e:
        return ticker, f"Error: {e}", "", "", ""


# Main code to read tickers and call the function
ticker_list_path = "stock_tickers.txt"
data_dir = "low_data"
output_file = "stock_data_lengths_model.csv"
start_date = "1993-11-06"
end_date = "2024-06-21"

# Make a directory
if __name__ == "__main__":
    os.makedirs(data_dir, exist_ok=True)

    with open(ticker_list_path, "r") as f, open(
        output_file, "w", encoding="utf-8"
    ) as csvfile:
        lines = f.readlines()
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Ticker", "Data Length", "First Date", "Last Date", "Rolling_STD_Dev"]
        )  # Write header
        for line in lines:
            ticker = line.strip("\n")
            print(f"Processing {ticker}...")
            result = download_stock_data(ticker, data_dir, start_date, end_date)
            writer.writerow(result)
            print(f"Data for {ticker} processed: {result[1]} rows")
