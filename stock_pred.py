from statistics import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import sklearn
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2022-02-01'

df = data.DataReader('AAPL','yahoo',start,end)
df = df.drop(['Adj Close'], axis=1)

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][0:int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu',return_sequences=True))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)

model.save('stock_pred.h5')