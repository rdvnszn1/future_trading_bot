from ta.momentum import rsi,stochrsi
from ta.trend import *
from ta.volatility import donchian_channel_pband
from ta.volatility import BollingerBands
import  numpy as np
import os
from ta.momentum import *
from ta.volume import *

def add_features(data):

    new_data=data.copy()
    indicator_bb = BollingerBands(close=data["close"], window=20, window_dev=2)
    bb_h = indicator_bb.bollinger_hband()
    bb_l = indicator_bb.bollinger_lband()
    new_data["b_perc"]=((new_data.close-bb_l)/(bb_h-bb_l))*100
    new_data["roc"]=ROCIndicator(new_data.close).roc()
    new_data["roc"]=WilliamsRIndicator(new_data.high,new_data.low,new_data.close).williams_r()
    new_data["RSI"]=rsi(new_data.close)

    new_data["RSI6"]=rsi(new_data.close,6)

    new_data["RSI2"]=rsi(new_data.close,2)

    new_data["CMF"]=chaikin_money_flow(new_data.high,new_data.low,new_data.close,new_data.volume)
    new_data["VWAP"]   = volume_weighted_average_price(new_data.high,new_data.low,new_data.close,new_data.volume)


    new_data=new_data.drop(columns=["open", "high", "low", "close", "volume"])

    return  new_data
