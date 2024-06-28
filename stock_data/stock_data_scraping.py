import csv
import os

import pandas as pd
import talib as ta
import yfinance as yf


def download_stock_data(ticker, data_dir, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
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
        df["Close_SMA5"] = ta.SMA(df["Close"], timeperiod=5)
        df["Close_SMA20"] = ta.SMA(df["Close"], timeperiod=20)
        df["Close_SMA60"] = ta.SMA(df["Close"], timeperiod=60)
        df["Close_SMA120"] = ta.SMA(df["Close"], timeperiod=120)
        df["Close_SMA180"] = ta.SMA(df["Close"], timeperiod=180)

        df["Open_SMA5"] = ta.SMA(df["Open"], timeperiod=5)
        df["Open_SMA20"] = ta.SMA(df["Open"], timeperiod=20)
        df["Open_SMA60"] = ta.SMA(df["Open"], timeperiod=60)
        df["Open_SMA120"] = ta.SMA(df["Open"], timeperiod=120)
        df["Open_SMA180"] = ta.SMA(df["Open"], timeperiod=180)

        df["High_SMA5"] = ta.SMA(df["High"], timeperiod=5)
        df["High_SMA20"] = ta.SMA(df["High"], timeperiod=20)
        df["High_SMA60"] = ta.SMA(df["High"], timeperiod=60)
        df["High_SMA120"] = ta.SMA(df["High"], timeperiod=120)
        df["High_SMA180"] = ta.SMA(df["High"], timeperiod=180)

        df["Low_SMA5"] = ta.SMA(df["Low"], timeperiod=5)
        df["Low_SMA20"] = ta.SMA(df["Low"], timeperiod=20)
        df["Low_SMA60"] = ta.SMA(df["Low"], timeperiod=60)
        df["Low_SMA120"] = ta.SMA(df["Low"], timeperiod=120)
        df["Low_SMA180"] = ta.SMA(df["Low"], timeperiod=180)

        df["Adj Close_SMA5"] = ta.SMA(df["Adj Close"], timeperiod=5)
        df["Adj Close_SMA20"] = ta.SMA(df["Adj Close"], timeperiod=20)
        df["Adj Close_SMA60"] = ta.SMA(df["Adj Close"], timeperiod=60)
        df["Adj Close_SMA120"] = ta.SMA(df["Adj Close"], timeperiod=120)
        df["Adj Close_SMA180"] = ta.SMA(df["Adj Close"], timeperiod=180)

        df["Close_EMA5"] = ta.EMA(df["Close"], timeperiod=5)
        df["Close_EMA20"] = ta.EMA(df["Close"], timeperiod=20)
        df["Close_EMA60"] = ta.EMA(df["Close"], timeperiod=60)
        df["Close_EMA120"] = ta.EMA(df["Close"], timeperiod=120)
        df["Close_EMA180"] = ta.EMA(df["Close"], timeperiod=180)

        df["Open_EMA5"] = ta.EMA(df["Open"], timeperiod=5)
        df["Open_EMA20"] = ta.EMA(df["Open"], timeperiod=20)
        df["Open_EMA60"] = ta.EMA(df["Open"], timeperiod=60)
        df["Open_EMA120"] = ta.EMA(df["Open"], timeperiod=120)
        df["Open_EMA180"] = ta.EMA(df["Open"], timeperiod=180)

        df["High_EMA5"] = ta.EMA(df["High"], timeperiod=5)
        df["High_EMA20"] = ta.EMA(df["High"], timeperiod=20)
        df["High_EMA60"] = ta.EMA(df["High"], timeperiod=60)
        df["High_EMA120"] = ta.EMA(df["High"], timeperiod=120)
        df["High_EMA180"] = ta.EMA(df["High"], timeperiod=180)

        df["Low_EMA5"] = ta.EMA(df["Low"], timeperiod=5)
        df["Low_EMA20"] = ta.EMA(df["Low"], timeperiod=20)
        df["Low_EMA60"] = ta.EMA(df["Low"], timeperiod=60)
        df["Low_EMA120"] = ta.EMA(df["Low"], timeperiod=120)
        df["Low_EMA180"] = ta.EMA(df["Low"], timeperiod=180)

        df["Adj Close_EMA5"] = ta.EMA(df["Adj Close"], timeperiod=5)
        df["Adj Close_EMA20"] = ta.EMA(df["Adj Close"], timeperiod=20)
        df["Adj Close_EMA60"] = ta.EMA(df["Adj Close"], timeperiod=60)
        df["Adj Close_EMA120"] = ta.EMA(df["Adj Close"], timeperiod=120)
        df["Adj Close_EMA180"] = ta.EMA(df["Adj Close"], timeperiod=180)

        # df["Close_WMA5"] = ta.WMA(df["Close"], timeperiod=5)
        # df["Close_WMA20"] = ta.WMA(df["Close"], timeperiod=20)
        # df["Close_WMA60"] = ta.WMA(df["Close"], timeperiod=60)
        # df["Close_WMA120"] = ta.WMA(df["Close"], timeperiod=120)
        # df["Close_WMA180"] = ta.WMA(df["Close"], timeperiod=180)

        # df["Open_WMA5"] = ta.WMA(df["Open"], timeperiod=5)
        # df["Open_WMA20"] = ta.WMA(df["Open"], timeperiod=20)
        # df["Open_WMA60"] = ta.WMA(df["Open"], timeperiod=60)
        # df["Open_WMA120"] = ta.WMA(df["Open"], timeperiod=120)
        # df["Open_WMA180"] = ta.WMA(df["Open"], timeperiod=180)

        # df["High_WMA5"] = ta.WMA(df["High"], timeperiod=5)
        # df["High_WMA20"] = ta.WMA(df["High"], timeperiod=20)
        # df["High_WMA60"] = ta.WMA(df["High"], timeperiod=60)
        # df["High_WMA120"] = ta.WMA(df["High"], timeperiod=120)
        # df["High_WMA180"] = ta.WMA(df["High"], timeperiod=180)

        # df["Low_WMA5"] = ta.WMA(df["Low"], timeperiod=5)
        # df["Low_WMA20"] = ta.WMA(df["Low"], timeperiod=20)
        # df["Low_WMA60"] = ta.WMA(df["Low"], timeperiod=60)
        # df["Low_WMA120"] = ta.WMA(df["Low"], timeperiod=120)
        # df["Low_WMA180"] = ta.WMA(df["Low"], timeperiod=180)

        # df["Adj Close_WMA5"] = ta.WMA(df["Adj Close"], timeperiod=5)
        # df["Adj Close_WMA20"] = ta.WMA(df["Adj Close"], timeperiod=20)
        # df["Adj Close_WMA60"] = ta.WMA(df["Adj Close"], timeperiod=60)
        # df["Adj Close_WMA120"] = ta.WMA(df["Adj Close"], timeperiod=120)
        # df["Adj Close_WMA180"] = ta.WMA(df["Adj Close"], timeperiod=180)



        df["RSI6"] = ta.RSI(df["Close"], timeperiod=6)#短期
        df["RSI14"] = ta.RSI(df["Close"], timeperiod=14)#一般使用14
        df["RSI24"] = ta.RSI(df["Close"], timeperiod=24)#長期


        df["MACD"], df["DIF"], df["OSC_Hist"] = ta.MACD(
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
        df["Daily_Return"] = df["Close"].pct_change()
        df["Daily_Change"] = df["Close"].diff()
        df["STD"] = df["Close"].std()

        # df["symbol"] = ticker

        # Drop rows with any NaN values
        # df.dropna(inplace=True)

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
ticker_list_path = "stock_tickers.txt"
data_dir = "stock_data_model"
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
        ["Ticker", "Data Length", "First Date", "Last Date"]
    )  # Write header
        for line in lines:
            ticker = line.strip("\n")
            print(f"Processing {ticker}...")
            result = download_stock_data(ticker, data_dir, start_date, end_date)
            writer.writerow(result)
            print(f"Data for {ticker} processed: {result[1]} rows")

