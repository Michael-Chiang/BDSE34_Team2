import numpy as np  # for handling potential division by zero
import pandas as pd
import yfinance as yf

# Read stock IDs from TXT file
with open("stock_id.txt", "r") as f:
    stock_ids = [line.strip() for line in f]


def calculate_financial_metrics(stock_id):
    try:
        stock = yf.Ticker(stock_id)
        pnl = stock.financials
        bs = stock.balancesheet
        fs = pd.concat([pnl, bs])
        fs = fs.T

        # Get most recent 4 years of data
        last_four_years = fs.iloc[-4:]  # Select last 4 rows (years)

        roe_roa_eps = {}
        for year in last_four_years.index:
            try:
                net_income = last_four_years.loc[year, "Net Income"]
                stockholders_equity = last_four_years.loc[year, "Stockholders Equity"]
                total_assets = last_four_years.loc[year, "Total Assets"]
                operating_income = last_four_years.loc[year, "Operating Income"]
                gross_profit = last_four_years.loc[year, "Gross Profit"]
                total_revenue = last_four_years.loc[year, "Total Revenue"]

                if total_revenue == 0:
                    operating_profit_margin = np.nan  # Assign np.nan if total_revenue is zero
                else:
                    operating_profit_margin = operating_income / total_revenue

                if total_revenue == 0:
                    net_profit_margin = np.nan  # Assign np.nan if total_revenue is zero
                else:
                    net_profit_margin = net_income / total_revenue

                gross_profit_margin = gross_profit / total_revenue
                roe = net_income / stockholders_equity
                roa = net_income / total_assets
                eps = last_four_years.loc[year, "Basic EPS"]

                roe_roa_eps[year] = {
                    "Gross_Profit_Margin": gross_profit_margin,
                    "Operating_Profit_Margin": operating_profit_margin,
                    "Net_Profit_Margin": net_profit_margin,
                    "ROE": roe,
                    "ROA": roa,
                    "EPS": eps
                }
            except KeyError:
                print(f"Error retrieving data for {stock_id} - Year: {year}")
                continue

        return roe_roa_eps

    except Exception as e:
        print(f"Error retrieving data for {stock_id}: {e}")
        return {}


# Download and calculate metrics for each stock and save to CSV
for stock_id in stock_ids:
    metrics = calculate_financial_metrics(stock_id)
    if metrics:
        df = pd.DataFrame(metrics).T
        df.index.name = "Date"

        # Sort DataFrame by index (dates) in descending order (optional)
        df = df.sort_index(ascending=False)  # Sort dates in descending order

        # Save DataFrame to CSV with informative filename
        filename = f"stock_index/{stock_id}_financial_metrics.csv"
        df.to_csv(filename)
        print(f"Financial metrics for {stock_id} saved to {filename}")
    else:
        print(f"No financial metrics retrieved for {stock_id}.")
