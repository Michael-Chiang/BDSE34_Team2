import pandas as pd
import yfinance as yf

stock_id = 'GOOG'
stock = yf.Ticker(stock_id)
pnl = stock.financials
bs = stock.balancesheet
fs = pd.concat([pnl,bs])
fs = fs.T
roe_roa_eps = {}
for year in fs.index:
    net_income = fs.loc[year, "Net Income"]
    stockholders_equity = fs.loc[year, "Stockholders Equity"]
    total_assets = fs.loc[year, "Total Assets"]
    operating_income = fs.loc[year, "Operating Income"]
    gross_profit = fs.loc[year, "Gross Profit"]
    total_revenue = fs.loc[year, "Total Revenue"]


    roe = net_income / stockholders_equity
    roa = net_income / total_assets
    eps = fs.loc[year, "Basic EPS"]
    gross_profit_margin = gross_profit / total_revenue
    operating_profit_margin = operating_income / total_revenue
    net_profit_margin = net_income / total_revenue
    
    roe_roa_eps[year] = {
        "Gross_Profit_Margin": gross_profit_margin,
        "Operating_Profit_Margin": operating_profit_margin,
        "Net_Profit_Margin": net_profit_margin,
        "ROE": roe,
        "ROA": roa,
        "EPS": eps
    }

# Print or display the results
print(roe_roa_eps)

# Convert dictionary to DataFrame with MultiIndex for clarity
multi_index = pd.MultiIndex.from_tuples([(year, stock_id) for year in fs.index],
                                         names=("Date", "Symbol"))

df = pd.DataFrame(roe_roa_eps).T
df.index = multi_index

df.index.name = "Date"
print(df)
# Save DataFrame to CSV file
df.to_csv(f"stock_index/{stock_id}_financial_metrics.csv")

print("Financial metrics saved to financial_metrics.csv")