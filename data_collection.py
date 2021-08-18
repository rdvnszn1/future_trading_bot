import requests
import datetime
import pandas as pd
import time

def collect_data(pair,timeframe="5m",limit=10000):
    file_name='data/{}_{}_data.csv'.format(pair,timeframe)
    get_btc_klines = requests.get("http://fapi.binance.com/fapi/v1/klines?symbol={}&interval={}&limit=1000".format(pair,timeframe)).json()

    date_list=[]
    open_list=[]
    high_list=[]
    low_list=[]
    close_list=[]
    volume_list=[]
    datetime_list=[]
    while True:
        for kline in get_btc_klines:
            human_time = datetime.datetime.fromtimestamp(kline[0] / 1000).isoformat()
            # print(human_time)
            kline.append(human_time)
            # print(kline)

            date_list.append(kline[0])
            open_list.append(float(kline[1]))
            high_list.append(float(kline[2]))
            low_list.append(float(kline[3]))
            close_list.append(float(kline[4]))
            volume_list.append(float(kline[5]))

            datetime_list.append(human_time)
        if len(date_list)>limit:
            break

        print(len(date_list))
        if len(date_list)%1000!=0:
            break
        last_kline = get_btc_klines[0]
        last_endTime = last_kline[0]
        get_btc_klines = requests.get(
            "http://fapi.binance.com/fapi/v1/klines?symbol={}&interval={}&limit=1000&endTime=".format(pair,timeframe) + str(
                last_endTime)).json()


    df=pd.DataFrame(index=date_list)
    df["open"]=open_list
    df["high"]=high_list
    df["low"]=low_list
    df["close"]=close_list
    df["volume"]=volume_list
    df["datetime"]=datetime_list

    df=df.sort_index()

    df.index=pd.to_datetime(df.datetime)
    del df["datetime"]

    # df.to_csv(file_name)

    return df


def take_data(pair,timeframe="5m",limit=500):

    current_time=time.time()*1000
    api_url = "https://fapi.binance.com/fapi/v1/klines?symbol={}&interval={}&limit={}".format(pair,timeframe,limit)
    json_data=requests.get(api_url).json()


    clean_json_data= [i[:6] for i in json_data if i[6]< current_time]
    dataframe = pd.DataFrame(clean_json_data)
    dataframe.columns = ["date", "open", "high", "low", "close", "volume"]
    dataframe = dataframe.set_index("date", drop=True)
    # dataframe.index = pd.to_datetime(dataframe.index, unit="ms")

    for c in dataframe.columns:
        dataframe[c] = dataframe[c].astype(float)


    dataframe = dataframe[~dataframe.index.duplicated()]

    return dataframe
