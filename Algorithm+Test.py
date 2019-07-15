import TestDataProcessing
import numpy as np
import pandas as pd

Processed_Data, Processed_Test = TestDataProcessing.PreprocessingTest()

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
  
Train_data = Processed_Data.sample(frac=0.7)
Test_data = Processed_Test

X_train = np.asarray(Train_data.loc[:, Train_data.columns != 'Sales'])
y_train = np.asarray(Train_data["Sales"])


X_test = np.asarray(Test_data.loc[:, Test_data.columns != 'Sales'])
y_test = np.asarray(Test_data["Sales"])

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, random_state=0, verbose=1)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

predictions1=pd.DataFrame(y_pred)
Test_y1=pd.DataFrame(y_test)

dfRes = pd.concat([predictions1,Test_y1],axis=1)
dfRes.columns=["pred","y"]
dfRes.drop(dfRes[dfRes.y == 0].index, axis=0, inplace=True)
print("RMSPE is", metric(np.array(dfRes.pred), np.array(dfRes.y)))
