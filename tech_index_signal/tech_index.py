import numpy as np
import pandas as pd

folder = 'stock_data_new'

for filename in os.listdir(folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(folder, filename)
        stock = pd.read_csv(filepath, index_col=['Date'], parse_dates=['Date'])
        Symbol = os.path.splitext(filename)[0]

        # 計算技術指標 (跟原本邏輯相同)
        stock['RSI14_over_heat'] = stock['RSI14'] > 70
        stock['RSI14_over_cold'] = stock['RSI14'] < 30

        stock['Close > SMA5'] = stock['Close'] > stock['Close_SMA5']
        stock['STD > 1.5'] = stock['Close'].rolling(10).std() > 1.5

        def crossover(over, down):
            a1 = over
            b1 = down
            a2 = a1.shift(1)
            b2 = b1.shift(1)
            crossover = (a1 > a2) & (a1 > b1) & (b2 > a2)
            return crossover

        def crossunder(down, over):
            a1 = down
            b1 = over
            a2 = a1.shift(1)
            b2 = b1.shift(1)
            crossunder = (a1 < a2) & (a1 < b1) & (b2 < a2)
            return crossunder

        stock['KD_Golden_Cross'] = crossover(stock['Stochastic_K'], stock['Stochastic_D'])
        stock['KD_Death_Cross'] = crossunder(stock['Stochastic_K'], stock['Stochastic_D'])

        low_range_kd = 25
        stock['Low_Range_D'] = stock['Stochastic_D'] < 25

        stock['Buy_In'] = (stock['KD_Golden_Cross'] & stock['Low_Range_D'])
        stock['Sold_Out'] = stock['KD_Death_Cross']

        stock['MACD_Golden_Cross'] = (stock['DIF'] > stock['MACD']) & (stock['DIF'].shift(1) < stock['MACD'].shift(1))
        stock['MACD_Death_Cross'] = (stock['DIF'] < stock['MACD']) & (stock['DIF'].shift(1) > stock['MACD'].shift(1))

        # 儲存結果為新檔 (檔名使用股票代碼)
        stock.to_csv(f'stock_data_new/{Symbol}.csv')
        
        print(f"{Symbol}處理完畢!")


  



  # 儲存結果為新檔 (檔名使用股票代碼)
  stock.to_csv(f'{stock_id}(new).csv')

print(f"{stock_id}處理完畢!")
