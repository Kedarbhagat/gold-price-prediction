import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""# **DATA PREPROCESSING**"""

dataset = pd.read_csv('gold_price_data.csv')

dataset.head()
date = dataset['Date']
date.size

dataset.info()

dataset.isnull().sum()

dataset.duplicated().sum()

dataset['Date'] = pd.to_datetime(dataset['Date'])

dataset.info()

dataset['year'] = dataset['Date'].dt.year
dataset['month'] = dataset['Date'].dt.month
dataset['day'] = dataset['Date'].dt.day
dataset['Quarter'] = dataset['Date'].dt.quarter

dataset

dataset = dataset.drop('Date', axis=1)

dataset

titles =list(dataset.columns)
titles

titles[1] , titles[-1] = titles[-1] , titles[1]
titles

dataset = dataset[titles]
dataset

"""PLOTTING"""

for column in dataset.select_dtypes(include=np.number):
  plt.figure()
  sns.boxplot(x=dataset[column])
  plt.title(f'Boxplot of {column}')
  plt.show()

# prompt: scatter plot for all feature vs gld

for column in dataset.select_dtypes(include=np.number):
  if column != 'GLD':
    plt.figure()
    plt.scatter(dataset[column], dataset['GLD'])
    plt.xlabel(column)
    plt.ylabel('GLD')
    plt.title(f'Scatter Plot of {column} vs GLD')
    plt.show()

dataset.select_dtypes(include ='number').corr()

sns.heatmap(dataset.select_dtypes(include ='number').corr(), annot=True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print(X.shape)

print(y)
print(y.shape)

"""# **DATASET SPLITING INTO TRAIN AND TEST**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_train)
print(X_train.shape)

print(y_train)
print(y_train.shape)

print(X_test)
print(X_test.shape)

print(y_test)
print(y_test.shape)

x_test_dataframe = pd.DataFrame(X_test ,columns=['SPX', 'Quarter', 'USO', 'SLV', 'EUR/USD', 'year', 'month', 'day'])
x_test_dataframe

"""# **MODEL** TRAINING

LINEAR REGRESSION
"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_linearRegression = regressor.predict(X_test)


from sklearn.metrics import r2_score
print(f"R2 score for linear Regression is : {r2_score(y_test,y_pred_linearRegression)}")

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred_linearRegression)
print("Mean Absolute Error:", mae)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred_linearRegression)
print("Mean Squared Error:", mse)

plt.figure(figsize=(10, 8))
plt.scatter(x_test_dataframe['year'], y_test, color='blue', s= 3, label='Actual GLD Price')
plt.scatter(x_test_dataframe['year'], y_pred_linearRegression, color='red',s=3 , label='Predicted GLD Price')
plt.xlabel('Year')
plt.ylabel('GLD Price')
plt.title('Trend Line Plot of GLD Price vs Year')
plt.legend()
plt.show()

print(np.concatenate((y_pred_linearRegression.reshape(len(y_pred_linearRegression),1), y_test.reshape(len(y_test),1)),1))

"""POLYNOMIAL REGRESSION"""

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
y_pred_polynomialRegression = lin_reg_2.predict(poly_reg.transform(X_test))
score_polynomialRegression=r2_score(y_test,y_pred_polynomialRegression)
score_polynomialRegression

print(np.concatenate((y_pred_polynomialRegression.reshape(len(y_pred_polynomialRegression),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import mean_absolute_error
mae_p = mean_absolute_error(y_test, y_pred_PolynomialRegression)
print("Mean Absolute Error:", mae_p)

from sklearn.metrics import mean_squared_error
mse_p = mean_squared_error(y_test, y_pred_PolynomialRegression)
print("Mean Squared Error:", mse_p)

plt.figure(figsize=(10, 8))
plt.scatter(x_test_dataframe['year'], y_test, color='blue', s= 3, label='Actual GLD Price')
plt.scatter(x_test_dataframe['year'], y_pred_PolynomialRegression, color='red',s=3 , label='Predicted GLD Price')
plt.xlabel('Year')
plt.ylabel('GLD Price')
plt.title('Trend Line Plot of GLD Price vs Year')
plt.legend()
plt.show()

"""SUPPORT VECTOR REGRESSION"""

#Support Vector Regression
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
y_train_scaled = sc.fit_transform(y_train.reshape(-1,1))
y_test_scaled = sc.transform(y_test.reshape(-1,1))

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = regressor.predict(X_test_scaled)
y_pred_svr = sc.inverse_transform(y_pred_scaled.reshape(-1,1))
score_SVR=r2_score(y_test,y_pred_svr)
score_SVR

print(np.concatenate((y_pred_svr.reshape(len(y_pred_svr),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
print(f"R2 score for Support Vector Regression is : {r2_score(y_test,y_pred_svr)}")

from sklearn.metrics import mean_absolute_error
mae_svr = mean_absolute_error(y_test, y_pred_svr)
print("Mean Absolute Error:", mae_svr)

from sklearn.metrics import mean_squared_error
mse_svr = mean_squared_error(y_test, y_pred_svr)
print("Mean Squared Error:", mse_svr)

plt.figure(figsize=(10, 8))
plt.scatter(x_test_dataframe['year'], y_test, color='blue', s= 3, label='Actual GLD Price')
plt.scatter(x_test_dataframe['year'], y_pred_svr, color='red',s=3 , label='Predicted GLD Price')
plt.xlabel('Year')
plt.ylabel('GLD Price')
plt.title('Trend Line Plot of GLD Price vs Year')
plt.legend()
plt.show()

"""DECISION TREE"""

#decision tree
from datetime import datetime
now = datetime.now()
quarter = (now.month - 1) // 3 + 1
year = now.year
month = now.month
day = now.day

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 6)
regressor.fit(X_train, y_train)
y_pred_decisionTree = regressor.predict(X_test)

score_decisionTree=r2_score(y_test,y_pred_decisionTree)
score_decisionTree

print(np.concatenate((y_pred_decisionTree.reshape(len(y_pred_decisionTree),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
print(f"R2 score for decision tree is : {r2_score(y_test,y_pred_decisionTree)}")

from sklearn.metrics import mean_absolute_error
mae_dt = mean_absolute_error(y_test, y_pred_decisionTree)
print("Mean Absolute Error:", mae_dt)

from sklearn.metrics import mean_squared_error
mse_dt = mean_squared_error(y_test, y_pred_decisionTree)
print("Mean Squared Error:", mse_dt)

plt.figure(figsize=(10, 8))
plt.scatter(x_test_dataframe['year'], y_test, color='blue', s= 3, label='Actual GLD Price')
plt.scatter(x_test_dataframe['year'], y_pred_decisionTree, color='red',s=3 , label='Predicted GLD Price')
plt.xlabel('Year')
plt.ylabel('GLD Price')
plt.title('Trend Line Plot of GLD Price vs Year')
plt.legend()
plt.show()

"""RANDOM FORREST"""

#random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred_randomForest = regressor.predict(X_test)
score_randomForest=r2_score(y_test,y_pred_randomForest)
score_randomForest

print(np.concatenate((y_pred_randomForest.reshape(len(y_pred_randomForest),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
print(f"R2 score for Random Forest is : {r2_score(y_test,y_pred_randomForest)}")

from sklearn.metrics import mean_absolute_error
mae_rf = mean_absolute_error(y_test, y_pred_randomForest)
print("Mean Absolute Error:", mae_rf)

from sklearn.metrics import mean_squared_error
mse_rf = mean_squared_error(y_test, y_pred_randomForest)
print("Mean Squared Error:", mse_rf)

plt.figure(figsize=(10, 8))
plt.scatter(x_test_dataframe['year'], y_test, color='blue', s= 3, label='Actual GLD Price')
plt.scatter(x_test_dataframe['year'], y_pred_randomForest, color='red',s=3 , label='Predicted GLD Price')
plt.xlabel('Year')
plt.ylabel('GLD Price')
plt.title('Trend Line Plot of GLD Price vs Year')
plt.legend()
plt.show()

print(score_linearRegression)
print(score_polynomialRegression)
print(score_SVR)
print(score_decisionTree)
print(score_randomForest)

from sklearn.model_selection import KFold, cross_val_score
def cross_validation(modelClass, *args, **kwargs):
    model = modelClass(*args, **kwargs)
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    print(f"Cross-Validation R² Scores for {modelClass}:", scores)
    print(f"Mean R² Score for {modelClass}:", np.mean(scores))

cross_validation(LinearRegression)
cross_validation(RandomForestRegressor, n_estimators=10, random_state=0)
cross_validation(DecisionTreeRegressor)
cross_validation(SVR)

output_list = [[score_linearRegression , mae , mse ] ,
               [score_polynomialRegression , mae_p ,mse_p],
               [score_SVR , mae_svr , mse_svr],
               [score_decisionTree ,mae_dt ,mse_dt],
               [score_randomForest , mae_rf , mse_rf]]
output_dataframe = pd.DataFrame(output_list, columns=['R2_score', 'MAE', 'MSE'])
output_dataframe.index = ['Linear Regression', 'Polynomial Regression', 'SVR', 'Decision Tree', 'Random Forest']
output_dataframe
"""R² Score: Indicates the proportion of variance explained by the model. The Random Forest model has the highest R² score (0.995821), explaining about 99.58% of the variance in gold prices, with all models above 0.90 indicating strong performance.

Mean Absolute Error (MAE): Measures the average prediction error without direction. The Random Forest model has the lowest MAE (0.971179), indicating its predictions are, on average, only $0.97 away from actual values.

Mean Squared Error (MSE): Penalizes larger errors more heavily. The Random Forest model has the lowest MSE (2.205451), showing fewer large errors compared to other models.
"""
print("Model with best performing score is Random Forest")