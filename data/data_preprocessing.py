import os
import pandas as pd
import talib

stock_names = ['AA', 'AXP', 'BAC', 'C', 'CSCO', 'DWDP', 'GM', 'HPQ', 'IBM', 'INTC', 'JPM', 'KFT', 'KO', 'MCD',
               'MMM', 'NKE', 'PGP', 'RTX', 'T', 'VZ']
raw_data_path = "../../data/Dow 30 1 min/"
raw_data_dest_path = "./raw_data/"
if not os.path.exists(raw_data_dest_path):
    os.makedirs(raw_data_dest_path)
using_TA = True


def convert_to_datetime(date_str, time_str):
    datetime_str = f"{date_str} {time_str}:00"
    datetime_obj = pd.to_datetime(datetime_str, format="%Y/%m/%d %H:%M:%S")
    return datetime_obj.strftime("%Y-%m-%d %H:%M:%S")


for stock_name in stock_names:
    date_list, time_list, open_list, high_list, low_list, close_list, volume_list = [], [], [], [], [], [], []
    with open(os.path.join(raw_data_path, f"{stock_name}.txt")) as f:
        lines = f.readlines()
        for line in lines:
            date, time, open_p, high_p, low_p, close_p, volume = line.strip().split(',')
            open_p, high_p, low_p, close_p, volume = float(open_p), float(high_p), float(low_p), float(close_p), float(
                volume)
            date_list.append(date)
            time_list.append(time)
            open_list.append(open_p)
            high_list.append(high_p)
            low_list.append(low_p)
            close_list.append(close_p)
            volume_list.append(volume)

    df = pd.DataFrame({
        "date": date_list,
        "time": time_list,
        "open": open_list,
        "high": high_list,
        "low": low_list,
        "close": close_list,
        "volume": volume_list
    })

    if using_TA:
        # MA
        df_close = df["close"]
        sma_5 = talib.SMA(df_close, timeperiod=5)
        sma_10 = talib.SMA(df_close, timeperiod=10)
        sma_30 = talib.SMA(df_close, timeperiod=30)
        dema = talib.DEMA(df_close, timeperiod=30)
        kama = talib.KAMA(df_close, timeperiod=30)
        wma = talib.WMA(df_close, timeperiod=30)
        trima = talib.TRIMA(df_close, timeperiod=30)
        # Momentum Indicators
        df_high = df["high"]
        df_low = df["low"]
        adx = talib.ADX(df_high, df_low, df_close)
        rsi = talib.RSI(df_close)
        cci = talib.CCI(df_high, df_low, df_close)
        macd = talib.MACD(df_close)
        mom = talib.MOM(df_close)
        roc = talib.ROC(df_close)
        # Volume Indicator
        df_volume = df["volume"]
        ad = talib.AD(df_high, df_low, df_close, df_volume)
        obv = talib.OBV(df_close, df_volume)

        ta_name_list = ['sma_5', 'sma_10', 'sma_30', 'dema', 'kama', 'wma', 'trima', 'adx', 'rsi', 'cci', 'macd[0]',
                        'mom', 'roc', 'ad', 'obv']
        ta_list = [sma_5, sma_10, sma_30, dema, kama, wma, trima, adx, rsi, cci, macd[0], mom, roc, ad, obv]
        for ta_name, ta in zip(ta_name_list, ta_list):
            df[ta_name] = ta
        df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.strftime('%Y/%m/%d')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M').dt.time
    start_time = pd.to_datetime('9:30', format='%H:%M').time()
    end_time = pd.to_datetime('16:00', format='%H:%M').time()
    time_mask = (df['time'] >= start_time) & (df['time'] <= end_time)
    df = df[time_mask]
    df['time'] = df['time'].apply(lambda x: x.strftime('%H:%M'))
    df['date'] = df.apply(lambda row: convert_to_datetime(row['date'], row['time']), axis=1)
    df = df.drop('time', axis=1)

    df.to_csv(os.path.join(raw_data_dest_path, f"{stock_name}.csv"), index=False)
