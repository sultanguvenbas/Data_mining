import math

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares
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
