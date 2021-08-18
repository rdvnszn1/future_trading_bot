import pandas as pd
from data_prep.features import add_features
from data_prep.labels import price_change
import numpy as np
from data_collection import collect_data

# xgboost_selection=['b_perc_1min', 'b_perc_5min', 'roc_1min', 'roc_5min', 'RSI_1min', 'RSI_5min', 'RSI6_1min', 'RSI6_5min', 'RSI2_1min', 'RSI2_5min', 'CMF_1min', 'CMF_5min', 'rsi6_corr_14_1min', 'rsi6_corr_14_5min', 'rsi12_corr_14_5min', 'stoch_1min', 'dcp_1min', 'dcp_5min', 'sma8_5min', 'sma13_5min', 'sma21_5min', 'sma34_1min', 'sma100_1min', 'sma100_5min', 'b_perc_1_1min', 'b_perc_1_5min', 'b_perc_diff_1_5min', 'b_perc_2_5min', 'b_perc_diff_2_5min', 'b_perc_3_5min', 'b_perc_diff_3_5min', 'b_perc_4_5min', 'b_perc_diff_4_5min', 'roc_1_1min', 'roc_diff_2_5min', 'roc_3_5min', 'roc_diff_3_5min', 'roc_diff_4_1min', 'RSI_1_1min', 'RSI_diff_1_5min', 'RSI_2_1min', 'RSI_2_5min', 'RSI_diff_2_5min', 'RSI_diff_3_5min', 'RSI_4_5min', 'RSI_diff_4_1min', 'RSI_diff_4_5min', 'RSI6_1_1min', 'RSI6_diff_1_5min', 'RSI6_diff_2_5min', 'RSI6_diff_3_5min', 'RSI6_diff_4_5min', 'RSI2_diff_1_5min', 'RSI2_3_5min', 'RSI2_diff_3_5min', 'RSI2_diff_4_5min', 'CMF_1_1min', 'CMF_diff_1_5min', 'CMF_2_1min', 'CMF_2_5min', 'CMF_diff_2_5min', 'CMF_3_1min', 'CMF_4_1min', 'CMF_diff_4_5min', 'VWAP_diff_1_1min', 'VWAP_diff_1_5min', 'VWAP_diff_2_1min', 'VWAP_diff_2_5min', 'VWAP_3_5min', 'VWAP_diff_3_1min', 'VWAP_4_5min', 'VWAP_diff_4_1min', 'rsi6_corr_14_1_1min', 'rsi6_corr_14_1_5min', 'rsi6_corr_14_diff_1_5min', 'rsi6_corr_14_2_5min', 'rsi6_corr_14_3_5min', 'rsi6_corr_14_4_1min', 'rsi12_corr_14_1_5min', 'rsi12_corr_14_2_5min', 'rsi12_corr_14_4_5min', 'stoch_1_1min', 'stoch_diff_1_5min', 'stoch_2_1min', 'stoch_diff_2_5min', 'stoch_3_5min', 'stoch_diff_3_5min', 'stoch_diff_4_5min', 'dcp_1_1min', 'dcp_diff_1_5min', 'dcp_diff_2_5min', 'dcp_diff_3_5min', 'dcp_diff_4_5min', 'sma8_1_5min', 'sma8_diff_1_1min', 'sma8_diff_1_5min', 'sma8_2_5min', 'sma8_diff_2_1min', 'sma8_diff_2_5min', 'sma8_diff_3_1min', 'sma8_diff_4_1min', 'sma8_diff_4_5min', 'sma13_diff_1_1min', 'sma13_diff_1_5min', 'sma13_diff_2_1min', 'sma13_diff_2_5min', 'sma13_diff_3_1min', 'sma13_diff_3_5min', 'sma13_diff_4_1min', 'sma21_diff_1_1min', 'sma21_diff_1_5min', 'sma21_diff_2_1min', 'sma21_diff_2_5min', 'sma21_diff_3_1min', 'sma21_diff_3_5min', 'sma21_4_1min', 'sma21_4_5min', 'sma21_diff_4_1min', 'sma21_diff_4_5min', 'sma34_1_1min', 'sma34_diff_1_1min', 'sma34_diff_1_5min', 'sma34_2_1min', 'sma34_diff_2_1min', 'sma34_diff_2_5min', 'sma34_3_1min', 'sma34_3_5min', 'sma34_diff_3_5min', 'sma34_4_1min', 'sma34_4_5min', 'sma34_diff_4_5min', 'sma100_1_5min', 'sma100_diff_1_1min', 'sma100_diff_1_5min', 'sma100_2_5min', 'sma100_diff_2_5min', 'sma100_3_5min', 'sma100_diff_3_1min', 'sma100_diff_3_5min', 'sma100_4_5min', 'sma100_diff_4_1min', 'sma100_diff_4_5min', 'ratio_trend_sma_8_1min', 'ratio_trend_sma_21_1min', 'ratio_trend_sma_50_1min', 'ratio_trend_sma_100_1min', 'ratio_trend_sma_100_5min', 'ratio_trend_sma_200_5min']

def feature_label_range(pair,data_point=500):
    data1min= collect_data(pair,"1m",data_point*1.1).iloc[-data_point:]
    data1min = data1min[~data1min.index.duplicated()]

    data_point_5m=int(data_point/4)
    data5min=collect_data(pair,"5m",data_point_5m).iloc[-data_point_5m:]
    data5min = data5min[~data5min.index.duplicated()]
    #
    # data15min = pd.read_csv("data/{}_15m_data.csv".format(pair), parse_dates=True, index_col="datetime").iloc[
    #            -data_point:]
    # data15min = data15min[~data15min.index.duplicated()]
    #
    # data1h = pd.read_csv("data/{}_1h_data.csv".format(pair), parse_dates=True, index_col="datetime").iloc[
    #             -data_point:]
    # data1h = data1h[~data1h.index.duplicated()]

    x_1min=add_features(data1min)

    x_5min=add_features(data5min)

    # x_15min=add_features(data15min)

    # x_1h=add_features(data1h)

    cols=x_1min.columns
    x=pd.DataFrame(index=x_1min.index)

    for c in cols:
        x[c+"_1min"]= x_1min[c]
        x[c+"_5min"]= x_5min[c]
        # x[c+"_15min"]= x_15min[c]
        # x[c+"_1h"]= x_1h[c]

    x=x.fillna(method="ffill")

    y=price_change(data1min)


    y=y[500:]
    x=x.iloc[500:]
    print(np.unique(y, return_counts=True))

    return x,y
