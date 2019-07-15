def PreprocessingTest():
    import pandas as pd
    import numpy as np
    kaggle_data=pd.read_csv('new_train.csv',index_col=0)
    kaggle_data1=pd.read_csv('store.csv')
    kaggle_dataX=pd.read_csv('test.csv',index_col=0)
    kaggle_data['StateHoliday'] = kaggle_data['StateHoliday'].map({0:0,'0': 0, 'a': 1,'b': 2,'c': 3})
    kaggle_data1['StoreType'] = kaggle_data1['StoreType'].map({'a': 1,'b': 2,'c': 3, 'd':4})

    kaggle_data['Date'] = pd.to_datetime(kaggle_data['Date'])
    kaggle_data['Month'] = kaggle_data['Date'].dt.month
    kaggle_data['Year'] = kaggle_data['Date'].dt.year

    merge_data=kaggle_data.merge(kaggle_data1,on="Store", how='inner')

    SelCol_data=merge_data.loc[:,['Sales','Store','DayOfWeek', 'Customers','Promo', 'StateHoliday', 'SchoolHoliday','StoreType','CompetitionDistance','Month','Year']]
    Final_data=SelCol_data.dropna()
    Final_data.Store = pd.Categorical(Final_data.Store)
    Final_data.DayOfWeek = pd.Categorical(Final_data.DayOfWeek)
    Final_data.StateHoliday = pd.Categorical(Final_data.StateHoliday)
    Final_data.SchoolHoliday = pd.Categorical(Final_data.SchoolHoliday)
    Final_data.StoreType = pd.Categorical(Final_data.StoreType)
    Final_data.Month = pd.Categorical(Final_data.Month)
    Final_data.Promo = pd.Categorical(Final_data.Promo)
    Final_data.Year = pd.Categorical(Final_data.Year)
    Final_data.Customers = pd.to_numeric(Final_data.Customers)

    Final_data.drop(Final_data[Final_data.Sales == 0].index, axis=0, inplace=True)

    Final_data=pd.DataFrame(Final_data)

    ###Test Data Processing

    kaggle_dataX['StateHoliday'] = kaggle_dataX['StateHoliday'].map({0: 0, '0': 0, 'a': 1, 'b': 2, 'c': 3})
    #kaggle_data1['StoreType'] = kaggle_data1['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})

    kaggle_dataX['Date'] = pd.to_datetime(kaggle_dataX['Date'])
    kaggle_dataX['Month'] = kaggle_dataX['Date'].dt.month
    kaggle_dataX['Year'] = kaggle_dataX['Date'].dt.year

    merge_dataX = kaggle_dataX.merge(kaggle_data1, on="Store", how='inner')

    SelCol_dataX = merge_dataX.loc[:,
                  ['Sales', 'Store', 'DayOfWeek', 'Customers', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType',
                   'CompetitionDistance', 'Month', 'Year']]
    Final_Test = SelCol_dataX.dropna()
    Final_Test.Store = pd.Categorical(Final_Test.Store)
    Final_Test.DayOfWeek = pd.Categorical(Final_Test.DayOfWeek)
    Final_Test.StateHoliday = pd.Categorical(Final_Test.StateHoliday)
    Final_Test.SchoolHoliday = pd.Categorical(Final_Test.SchoolHoliday)
    Final_Test.StoreType = pd.Categorical(Final_Test.StoreType)
    Final_Test.Month = pd.Categorical(Final_Test.Month)
    Final_Test.Promo = pd.Categorical(Final_Test.Promo)
    Final_Test.Year = pd.Categorical(Final_Test.Year)
    Final_Test.Customers = pd.to_numeric(Final_Test.Customers)

    Final_Test.drop(Final_Test[Final_Test.Sales == 0].index, axis=0, inplace=True)

    Final_Test = pd.DataFrame(Final_Test)

    return Final_data, Final_Test


