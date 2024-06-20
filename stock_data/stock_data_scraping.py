import csv
import os

import pandas as pd
import talib as ta
import yfinance as yf


def download_stock_data(ticker, data_dir, start_date):
    try:
        df = yf.download(ticker, start=start_date)
        if df.empty:
            return ticker, 0, "", ""

        # Calculate technical indicators

        """
        SMA(Simple Moving Average)
        EMA(Exponential Moving Average)
        RSI(Relative Strength Index)
        MACD(Moving Average Convergence Divergence)
        Bollinger Bands
        Stochastic Oscillator (KD Inndicators)
        Williams_%R
        ATR(Average True Rage)
        MFI(Money Flow Index)
        OBV(On Balance Volume)
        Momentum
        ROC(Rate of Change)
        ADX(Average directional movement index)
        DMI(Directional Movement Index)
        """
        df["SMA5"] = ta.SMA(df["Close"], timeperiod=5)
        df["SMA30"] = ta.SMA(df["Close"], timeperiod=30)
        df["SMA60"] = ta.SMA(df["Close"], timeperiod=60)
        df["SMA180"] = ta.SMA(df["Close"], timeperiod=180)

        df["EMA5"] = ta.EMA(df["Close"], timeperiod=5)
        df["EMA30"] = ta.EMA(df["Close"], timeperiod=30)
        df["EMA60"] = ta.EMA(df["Close"], timeperiod=60)
        df["EMA180"] = ta.EMA(df["Close"], timeperiod=180)

        df["WMA5"] = ta.WMA(df["Close"], timeperiod=5)
        df["WMA30"] = ta.WMA(df["Close"], timeperiod=30)
        df["WMA60"] = ta.WMA(df["Close"], timeperiod=60)
        df["WMA180"] = ta.WMA(df["Close"], timeperiod=180)

        df["RSI"] = ta.RSI(df["Close"], timeperiod=180)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = ta.MACD(
            df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["Upper_Band"], df["Middle_Band"], df["Lower_Band"] = ta.BBANDS(
            df["Close"], timeperiod=20
        )
        df["Stochastic_K"], df["Stochastic_D"] = ta.STOCH(
            df["High"],
            df["Low"],
            df["Close"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )
        df["Williams_R"] = ta.WILLR(df["High"], df["Low"], df["Close"], timeperiod=14)
        df["ATR"] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
        df["MFI"] = ta.MFI(
            df["High"], df["Low"], df["Close"], df["Volume"], timeperiod=14
        )
        df["OBV"] = ta.OBV(df["Close"], df["Volume"])
        df["Momentum"] = ta.MOM(df["Close"], timeperiod=10)
        df["ROC"] = ta.ROC(df["Close"], timeperiod=10)
        df["ADX"] = ta.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
        df["PLUS_DMI"] = ta.PLUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)
        df["MINUS_DMI"] = ta.MINUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)

        # Drop rows with any NaN values
        df.dropna(inplace=True)

        if df.empty:
            return ticker, 0, "", ""

        # Save data with indicators
        df.to_csv(f"{data_dir}/{ticker}.csv")

        first_date = df.index[0].strftime("%Y-%m-%d")
        last_date = df.index[-1].strftime("%Y-%m-%d")
        return ticker, len(df), first_date, last_date
    except Exception as e:
        return ticker, f"Error: {e}", "", ""


# Main code to read tickers and call the function
ticker_list_path = "stock_stock.txt"
data_dir = "stock_data_model"
output_file = "stock_data_lengths_model.csv"
start_date = "1980-06-01"

# Make a directory
if __name__ == "__main__":
    os.makedirs(data_dir, exist_ok=True)

    with open(ticker_list_path, "r") as f, open(
    output_file, "w", encoding="utf-8"
) as csvfile:
        lines = f.readlines()
        writer = csv.writer(csvfile)
        writer.writerow(
        ["Ticker", "Data Length", "First Date", "Last Date"]
    )  # Write header
        for line in lines:
            ticker = line.strip("\n")
            print(f"Processing {ticker}...")
            result = download_stock_data(ticker, data_dir, start_date)
            writer.writerow(result)
            print(f"Data for {ticker} processed: {result[1]} rows")
