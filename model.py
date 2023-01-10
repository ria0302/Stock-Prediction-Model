import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

# download dataset from yfinance
ticker = yf.Ticker('AAPL')
apple_yf = yf.download('AAPL', start="2020-01-01", end="2023-09-01", progress=False)
print(apple_yf.shape)

# preprocessing data
print(apple_yf.info())
# no null values present
# feature scaling i.e standardising
y_pred = pd.get_dummies(apple_yf["Adj Close"])
apple_yf.drop(["Close", "Adj Close"], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(apple_yf, y_pred, random_state=0)
StandardScaler().fit(X_train)
StandardScaler().fit(y_train)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

# making the model with LSTM
model = Sequential()
# in the input layer, the shape of dataset must be given
print(X_train.shape)
model.add(LSTM(X_train.shape[1], input_shape=(X_train.shape[1], 1), name="Layer1"))
model.add(Dense(X_train.shape[1], name="Layer2"))
model.add(Dense(1, name="Output"))

# compiling
model.compile(optimizer='RMSprop', loss="mean_squared_error", metrics=["accuracy"])

# evaluating the model
model.fit(X_train, y_train, batch_size=50, epochs=50, verbose=0)
accuracy = model.evaluate(X_test, y_test)
print("The final loss is {} and the accuracy achieved is {}".format(accuracy[0], accuracy[1]))
