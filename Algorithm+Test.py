import TestDataProcessing
import numpy as np
import pandas as pd

Processed_Data, Processed_Test = TestDataProcessing.PreprocessingTest()

Train_data = Processed_Data.sample(frac=0.7)
Test_data = Processed_Test
y_train = Train_data["Sales"].values
X_train = Train_data.drop(columns=["Sales"])
assert not 'Sales' in X_train.columns

X_train1 = X_train.values

y_test = Test_data["Sales"].values
X_test = Test_data.drop(columns=["Sales"])
X_test1 = X_test.values
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, random_state=0, verbose=1)
regressor.fit(X_train1,y_train)
y_pred = regressor.predict(X_test1)
predictions1=pd.DataFrame(y_pred)
Test_y1 = pd.DataFrame(y_test)


dfRes = pd.concat([predictions1,Test_y1],axis=1)
dfRes.columns = ["pred","y"]
dfRes.drop(dfRes[dfRes.y == 0].index, axis=0, inplace=True)
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
# import pdb ; pdb.set_trace()
print("RMSPE is",metric(dfRes.pred.values,dfRes.y.values),"%")
print("Same RMSPE is ",100*(sum(((dfRes.pred-dfRes.y)/dfRes.y)**2)/len(dfRes.pred))**0.5,"%")
