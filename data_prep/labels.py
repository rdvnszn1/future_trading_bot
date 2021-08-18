import numpy as np
def price_change(df):
    data=df.copy()
    data["change"]=data["close"].pct_change()
    data["change_lag"]=data["change"].shift(-1)
    data["change_lag"]=data["change_lag"].fillna(0)
    data.loc[:,"y_change"]=0
    data.loc[( data["change_lag"] <  np.percentile(data["change_lag"],33)),"y_change"]=-1
    data.loc[( data["change_lag"] >  np.percentile(data["change_lag"],66)),"y_change"]=1
    return data["y_change"].values