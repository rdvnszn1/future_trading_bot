
def adjust_data(X, y, split=[0.6,0.2,0.2]):
    # count the number of each label

    cut_point1= int(split[0]*y.shape[0])
    cut_point2= int(split[1]*y.shape[0])+cut_point1
    # cut_point3= int(split[2]*y.shape[0])+cut_point2

    X_train, y_train  = X[:cut_point1], y[:cut_point1]
    X_test, y_test = X[cut_point1:cut_point2], y[cut_point1:cut_point2]
    X_trade,y_trade =  X[cut_point2:], y[cut_point2:]

    return X_train, X_test, y_train, y_test ,X_trade,y_trade
