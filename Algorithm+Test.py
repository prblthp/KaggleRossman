import TestDataProcessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Processed_Data, Processed_Test = TestDataProcessing.PreprocessingTest()

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

def plot_feature_importances(rf, cols):
    importances = pd.DataFrame()
    importances.loc[:, 'importances'] = rf.feature_importances_
    importances.loc[:, 'features'] = cols
    importances.sort_values('importances', inplace=True)
    f, a = plt.subplots()
    importances.plot(ax=a, kind='bar', x='features', y='importances')
    plt.gcf().subplots_adjust(bottom=0.3)
    f.savefig('importances.png')

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

cols = Train_data.loc[:, Train_data.columns != 'Sales'].columns
plot_feature_importances(regressor, cols)

print("RMSPE is", metric(y_pred, y_test))

cols = Train_data.loc[:, Train_data.columns != 'Sales'].columns
plot_feature_importances(regressor, cols)
