from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from data_prep.combine_tf_feature import *
from data_prep.helpers import *

pair_list=["BTCUSDT","ETHUSDT","BNBUSDT","FILUSDT","ADAUSDT","XRPUSDT","DOGEUSDT","ATOMUSDT","WAVESUSDT","KNCUSDT",
           "BATUSDT","SCUSDT","DGBUSDT","STORJUSDT","LINKUSDT","DGBUSDT","LRCUSDT","CVCUSDT","ETCUSDT"]

threshold_list=[0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
data_point=50000
result_list=[]

if __name__ == '__main__':

    for pair in pair_list:

        print("*"*100)
        result_dict = {}

        result_dict["pair"]=pair
        print(pair)
        X, y = feature_label_range(pair,data_point)

        X_train, X_test, y_train, y_test, X_trade, y_trade = adjust_data(X,y,[0.60,0.20,0.20])

        print(np.unique(y, return_counts=True))
        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_test, return_counts=True))
        print(np.unique(y_trade, return_counts=True))

        # xgb_model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)

        xgb_model = XGBClassifier(learning_rate=0.1, gamma=1.5, max_depth=2, colsample_bytree=1.0, subsample=0.6,
                                        reg_alpha=0, reg_lambda=1, min_child_weight=5, n_estimators=100)

        #train again
        xgb_model.fit(X_train, y_train)

        # make predictions for test data
        y_pred_with_prob = pd.DataFrame(xgb_model.predict_proba(X_test))
        y_pred_with_prob.columns = xgb_model.classes_
        # y_pred_with_prob.index=y_test.index
        y_pred = xgb_model.predict(X_test)
        # evaluate predictions
        accuracy = accuracy_score(y_test, y_pred)
        print("TEST Accuracy: %.2f%%" % (accuracy * 100.0))
        result_dict["test_accuracy"]=accuracy

        print("-" * 100)
        # evaluate predictions
        accuracy_trade = accuracy_score(y_trade, xgb_model.predict(X_trade))
        print("TRADE Accuracy: %.2f%%" % (accuracy_trade * 100.0))
        result_dict["accuracy_trade"]=accuracy_trade

        print("-" * 100)

        # PROB TEST
        predicted = xgb_model.predict(X_test)
        probabs = xgb_model.predict_proba(X_test)
        probabs_df = pd.DataFrame(probabs)
        probabs_df.columns = xgb_model.classes_
        probabs_df["true_labels"] = y_test
        probabs_df["predicted"] = predicted

        accuracy_score(probabs_df.true_labels, probabs_df.predicted)
        probabs_df["max_value"] = probabs_df[probabs_df.columns[:3]].max(axis=1)

        for threshold in threshold_list:

            threshold_df = probabs_df[probabs_df["max_value"] > threshold].copy()
            threshold_df=threshold_df[threshold_df["predicted"]!=0]
            print("TEST threshold accuracy: ", accuracy_score(threshold_df.true_labels, threshold_df.predicted), "with threshold: ",
                  threshold)
            print(len(threshold_df))

            result_dict["test_"+str(threshold)] = accuracy_score(threshold_df.true_labels, threshold_df.predicted)
            result_dict["test_len_"+str(threshold)] = len(threshold_df)

        # PROB TRADE
        predicted = xgb_model.predict(X_trade)
        probabs = xgb_model.predict_proba(X_trade)
        probabs_df = pd.DataFrame(probabs)
        probabs_df.columns = xgb_model.classes_
        probabs_df["true_labels"] = y_trade
        probabs_df["predicted"] = predicted

        accuracy_score(probabs_df.true_labels, probabs_df.predicted)
        probabs_df["max_value"] = probabs_df[probabs_df.columns[:3]].max(axis=1)

        for threshold in threshold_list:
            threshold_df = probabs_df[probabs_df["max_value"] > threshold].copy()
            threshold_df=threshold_df[threshold_df["predicted"]!=0]

            print("TRADE threshold accuracy: ", accuracy_score(threshold_df.true_labels, threshold_df.predicted),
                  "with threshold: ",
                  threshold)
            print(len(threshold_df))
            result_dict["trade"+str(threshold)] = accuracy_score(threshold_df.true_labels, threshold_df.predicted)
            result_dict["trade_len_"+str(threshold)] = len(threshold_df)

        result_list.append(result_dict)
        print(result_list)
        print("*"*100)



pd.DataFrame(result_list).to_excel("results.xlsx")

