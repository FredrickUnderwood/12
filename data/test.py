import pandas as pd

df = pd.read_csv("../data/news_data/sentiment_data.csv", encoding="utf-8")


def convert_to_datetime(date_str, time_str):
    if time_str.startswith('24'):
        time_str = '00' + time_str[2:]
        date_str = pd.to_datetime(date_str) + pd.Timedelta(days=1)
        datetime_str = f"{date_str.strftime('%Y/%m/%d')} {time_str}"
    else:
        datetime_str = f"{date_str} {time_str}"

    datetime_obj = pd.to_datetime(datetime_str, format="%Y/%m/%d %H:%M:%S")
    return datetime_obj
df['date'] = df.apply(lambda row: convert_to_datetime(row['date'], row['time']), axis=1)
df = df.drop('time', axis=1)
df.to_csv("../data/news_data/sentiment_data.csv", index=False)

