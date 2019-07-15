import TestDataProcessing
import numpy as np
import pandas as pd

Processed_Data, Processed_Test = TestDataProcessing.PreprocessingTest()

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
#print(predictions1)
Test_y1=pd.DataFrame(y_test)

print(Test_y1)
print(predictions1)
dfRes = pd.concat([predictions1,Test_y1],axis=1)
dfRes.columns=["pred","y"]
#print(dfRes.head)
#dfRes1=dfRes[dfRes.iloc[:, 1] != 0]
#print(sum(dfRes["Sales"]==0))
dfRes.drop(dfRes[dfRes.y == 0].index, axis=0, inplace=True)
print("RMSPE is",(sum(((dfRes.pred-dfRes.y)/dfRes.y)**2))/len(dfRes.pred)**0.5)
