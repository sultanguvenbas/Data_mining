import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Reading dataset file
filename = 'forestfires.csv'
df = pd.read_csv(filename)

#Taking temp	RH	wind	rain from end
data = df[df.columns[-5:]].values
np.random.shuffle(data)
X = data[:, :-1]  # temp	RH	wind	rain
y = data[:, -1]  # area

# Splitting train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X)
print(y)
