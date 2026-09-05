import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("insurance_data.csv")
df.head()

plt.scatter(df.age, df.bought_insurance)
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df[['age']], df.bought_insurance,train_size=0.8)

X_test

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

X_test

Y_prediction = model.predict(X_test)
model.predict_proba(X_test)
model.score(X_test, Y_test)

Y_prediction
X_test
model.coef_
model.intercept_

import math
def sigma(x):
    return 1 / (1+math.exp(-x))

def prediction_func(age):
    z = 0.042 * age - 1.53
    y = sigma(z)
    return y

age = 35
print(prediction_func(age))

age = 43
print(prediction_func(age))