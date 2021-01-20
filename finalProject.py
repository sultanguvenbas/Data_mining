import math

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")

# Reading dataset file

filename = 'forestfires.csv'
df = pd.read_csv(filename)

#Taking temp,RH,wind,rain and area
data = df[df.columns[-5:]].values
np.random.shuffle(data)
X = data[:, :-1]  # temp	RH	wind	rain
y = data[:, -1]  # area

# Splitting train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#print(X)
#print(y)

def rmse(y_true, y_predict):  # Root mean Square Error
    retval = mean_squared_error(y_true, y_predict)
    return math.sqrt(retval)

# Defining Levenberg-Marquardt Model


def model(x, a, b, c, d, e):
    # 0 -> temp
    # 1 -> RH
    # 2 -> wind
    # 3 -> rain

    top = (x[:, 0] * a) ** 2 + (x[:, 2] * c) ** 2
    bottom = (x[:, 1] * b) ** 2 + (x[:, 3] * d) ** 2
    return (top / bottom) + e


def func(params, xdata, ydata):
    return ydata - model(xdata, *params)


# Activation alternatives: 'identity', 'logistic', 'tanh', 'relu'
print('MLPRegressor Model')
mdl = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=5000, activation='relu',
                   learning_rate='constant', learning_rate_init=0.001)
mdl.fit(X_train, y_train)

# Printing RMSE of training and test dataset for MLP Model

y_predict = mdl.predict(X_train)
print('Training set rmse: %.3f' % rmse(y_train, y_predict))

y_predict = mdl.predict(X_test)
print('Test set rmse: %.3f' % rmse(y_test, y_predict))

print()
x0 = np.ones(5)  # taking ones
print('Least Squares Model with Levenberg-Marquardt algorithm')

result = least_squares(func, x0, args=(X_train, y_train), method='lm')

params = result.x
# Printing RMSE of training and test dataset for MLP Model Levenberg-Marquardt

y_predict = model(X_train, *params)
print('Training set rmse: %.3f' % rmse(y_train, y_predict))

y_predict = model(X_test, *params)
print('Test set rmse: %.3f' % rmse(y_test, y_predict))

def histogram():

    data2 = pd.read_csv(filename)
    months = data2[data2.columns[2]].values
    area = data2[data2.columns[-1]].values
    monthDict = {}

    for monthName in range(len(months)):
        if months[monthName] not in monthDict:
            monthDict[months[monthName]] = 0

    for areaindex in range(len(area)):
        monthDict[months[areaindex]] += area[areaindex]
    #print(monthDict)

    monthArray = ["jan","feb","mar","apr","may","jun","jul",
                  "aug","sep","oct","nov","dec"]

    arr=[]
    arrValue=[]

    for checkMont in monthArray:
        for keys, values in monthDict.items():
            if checkMont == keys:
                arr.append(keys)
                arrValue.append(values)

    #print(monthDict)
    #print(arr)
    #print(arrValue)

    plt.xlabel("Months")
    plt.ylabel("Total Burned Area (in ha)")
    plt.title("Histogram")
    plt.bar(arr,arrValue)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.show()

histogram()
