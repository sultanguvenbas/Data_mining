import math

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neural_network import MLPRegressor
import warnings

warnings.filterwarnings("ignore")

# Reading dataset file

filename = 'forestfires.csv'
df = pd.read_csv(filename)

# Taking temp,RH,wind,rain and area
data = df[df.columns[-5:]].values
np.random.shuffle(data)
X = data[:, :-1]  # temp	RH	wind	rain
y = data[:, -1]  # area

# Splitting train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# print(X)
# print(y)

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
mdl = MLPRegressor(hidden_layer_sizes=(16, 8, 4), max_iter=5000, activation='relu',
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
    data2 = pd.read_csv(filename)  # reading data-set again
    months = data2[data2.columns[2]].values  # taking months from data-set
    area = data2[data2.columns[-1]].values  # taking area from data-set

    # creating dictionary value and adding items inside of
    monthDict = {}
    for monthName in range(len(months)):
        if months[monthName] not in monthDict:
            monthDict[months[monthName]] = 0

    # sum up area for every months
    for areaindex in range(len(area)):
        monthDict[months[areaindex]] += area[areaindex]
    # print(monthDict)

    monthArray = ["jan", "feb", "mar", "apr", "may", "jun", "jul",
                  "aug", "sep", "oct", "nov", "dec"]

    arr = []
    arrValue = []

    # splitting the dictionary as a key and value
    for checkMont in monthArray:
        for keys, values in monthDict.items():
            if checkMont == keys:
                arr.append(keys)
                arrValue.append(values)

    # print(monthDict)
    # print(arr)
    # print(arrValue)

    # drawing the histogram
    plt.xlabel("Months")
    plt.ylabel("Total Burned Area (in ha)")
    plt.title("Histogram")
    plt.bar(arr, arrValue)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.show()


histogram()


def heatmeapFunc():
    copyDf = df.copy()  # copy it to not lose original data-set
    xCor = copyDf[copyDf.columns[0]].values  # taking all x coordinates
    yCor = copyDf[copyDf.columns[1]].values  # taking all y coordinates
    temp = copyDf[copyDf.columns[8]].values  # taking all temp

    # to count reapeted  coordinates there are (e.g 7-5 repeated 11 times)
    xYCorDict = {}
    averageTemp = {}  # put temp average of every coordinates
    for xY in range(len(xCor)):
        # define their keys (e.g 75,99)
        addToKey = str(xCor[xY]) + str(yCor[xY])
        if addToKey not in xYCorDict:  # to prevent redundancy
            xYCorDict[addToKey] = 0  # define them as 0 for now
            averageTemp[addToKey] = 0  # define them as 0 for now

    for xYCount in range(len(xCor)):
        addToKey = str(xCor[xYCount]) + str(yCor[xYCount])  # count repeated
        xYCorDict[addToKey] += 1  # add 1 for every repetation

    for tempCount in range(len(xCor)):
        addToKey = str(xCor[tempCount]) + str(yCor[tempCount])  # take the id
        averageTemp[addToKey] += temp[tempCount]  # sum up all temp

    for keys, values in xYCorDict.items():
        averageTemp[keys] = averageTemp[keys] / xYCorDict[keys]  # divided it

    arrayTemp = []  # put id and its average temp

    for keys, values in averageTemp.items():
        arrayTemp.append([keys, values])

    heatMapArray = []  # heat map array 10X10 array

    # since we know the X and Y coordinates is upto 9 we run it 10 times to create heatmap
    for xCordinate in range(10):
        addToHeatmap = []
        for yCordinate in range(10):
            addToHeatmap.append(0)
        heatMapArray.append(addToHeatmap)

    for i in range(len(arrayTemp)):
        heatMapArray[int(arrayTemp[i][0][0])][int(arrayTemp[i][0][1])] = float(arrayTemp[i][1])

    # for keys, value in averageTemp.items():
    # print(keys, value)
    heatMap = sb.heatmap(heatMapArray, annot=True, cmap='Reds')
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.title('Average Temperature In X,Y Coordinate (C)')
    heatMap.invert_yaxis()
    plt.show()


heatmeapFunc()
