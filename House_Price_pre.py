from typing import no_type_check
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
#read file
train_path='C:/Users/haoni/Downloads/home-data-for-ml-course/train.csv'
test_path='C:/Users/haoni/Downloads/home-data-for-ml-course/test.csv'
train=pd.read_csv(train_path)
test=pd.read_csv(test_path)
y=train.SalePrice
corr = train.corr()
corr.style.background_gradient(cmap='coolwarm')
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#plt.show()
"""
plt.scatter(train.TotalBsmtSF,y)
outliers=((train['GrLivArea']>4000) & (y<300000))|((train['OverallCond']==2) & (y>300000))
train.drop(train[outliers].index,inplace=True)
y.drop(y[outliers].index,inplace=True)
plt.show()"""
features = ["YearBuilt","YearRemodAdd","TotalBsmtSF","1stFlrSF","GrLivArea","GarageArea"]

X= train[features]
print(X.describe())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
rf_model_on_full_data = RandomForestRegressor()
rf_model_on_full_data.fit(X, y)
test_X = test[features]
imputer = SimpleImputer(strategy="median")
test_X=imputer.fit_transform(test_X)
print(np.where(np.isnan(test_X)))
test_preds = rf_model_on_full_data.predict(test_X)
output = pd.DataFrame({'Id': test.Id,'SalePrice': test_preds})
output.to_csv('C:/Users/haoni/Downloads/home-data-for-ml-course/submission.csv', index=False)
print("Successfully Predicted the House Price!")