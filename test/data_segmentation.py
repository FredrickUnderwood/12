import pandas as pd
import os


root_path = "../data/raw_data/"
attention_root_path = "../data/attention_data/"
if not os.path.exists(attention_root_path):
    os.makedirs(attention_root_path)

stock_names = ['AA', 'AXP', 'BAC', 'C', 'CSCO', 'DWDP', 'GM', 'HPQ', 'IBM', 'INTC', 'JPM', 'KFT', 'KO', 'MCD',
               'MMM', 'NKE', 'PGP', 'RTX', 'T', 'VZ']
def convert_to_datetime(date_str, time_str):
    datetime_str = f"{date_str} {time_str}:00"
    datetime_obj = pd.to_datetime(datetime_str, format="%Y/%m/%d %H:%M:%S")
    return datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
steps = 100
for stock in stock_names:
    stock_path = f"{stock}/"
    if not os.path.exists(os.path.join(attention_root_path, stock_path)):
        os.makedirs(os.path.join(attention_root_path, stock_path))
    df = pd.read_csv(os.path.join(root_path, f"{stock}.csv"))
    date_list = df["date"].unique()
    df_temp = []
    for i, date_ in enumerate(date_list):
        df_new = df[df["date"] == date_]
        df_new["date"] = df_new.apply(lambda row: convert_to_datetime(row['date'], row['time']), axis=1)
        df_final = df_new[['date', 'open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 'sma_30', 'dema', 'kama', 'wma', 'trima', 'adx', 'rsi', 'cci', 'macd[0]', 'mom', 'roc', 'ad', 'obv']]
        df_temp.append(df_final)
        if i % steps == 0:
            df_final_ = pd.concat(df_temp, ignore_index=True)
            df_final_.to_csv(os.path.join(attention_root_path, stock_path, f"{i // steps}.csv"), index=False)
            df_temp = []
        if i > 500:
            break
    break
