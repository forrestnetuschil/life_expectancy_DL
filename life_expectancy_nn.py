import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('/Life_Expectancy_Data.csv')

df.isnull().sum()

#cleaning and renaming columns
df.rename(columns = {'Country' : 'country',
	'Year' : 'year', 'Status' : 'status', 
	'Life expectancy ' : 'life expectancy', 'Adult Mortality' : 'adult mortality',
	'Alcohol' : 'alcohol', 'Measles ' : 'measles',
	' BMI ' : 'BMI', 'under-five deaths ' : 'under-five deaths',
	'Diphtheria ' : 'diphtheria', ' HIV/AIDS' : 'HIV/AIDS'}, inplace = True)

#finding the corralation between columns and rows
df.corr()

#setting x and y variables
x = df[['adult mortality', 'infant deaths', 'alcohol',
'percentage expenditure', 'Hepatitis B', 'measles',
'Polio', 'Total expenditure', 'diphtheria', 'HIV/AIDS',
'GDP', 'Population', 'Schooling']]
y = df['life expectancy']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.25)

x_train.shape
x_test.shape

life_exp = LinearRegression()

life_exp.fit(x_train, y_train)

life_exp.score(x_test, y_test)

life_exp.score(x_train, y_train)

y_pred = life_exp.predict(x_test)

metrics.mean_absolute_error(y_test, y_pred)

metrics.mean_squared_error(y_test, y_pred)

print(life_exp.intercept_)
print(life_exp.coef_)

np.exp(life_exp.coef_)

mlpr = MLPRegressor(hidden_layer_sizes = (6,2), max_iter = 20000, random_state = 42)

mlpr.fit(x_train, y_train)

mlpr.score(x_train, y_train)