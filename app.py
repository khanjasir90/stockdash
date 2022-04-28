from cProfile import label
import streamlit as st
from plotly import graph_objs as go
import yfinance as yf
from datetime import date
from sktime.forecasting.fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from statistics import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as dataa
import sklearn
from sklearn.preprocessing import MinMaxScaler

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction Application')


stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
title = st.text_input('Stock Ticker Name', 'MSFT')
st.write('You Selected ', title)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(title)
data_load_state.text('Loading data... done!')

st.subheader(title+' Stock Data')
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

df = data
st.subheader('Closing Price vs Time Chart with 100MA')
m100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(m100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
fig = plt.figure(figsize=(12,6))
m100=df.Close.rolling(100).mean()
m200=df.Close.rolling(200).mean()
plt.plot(m100,'r')
plt.plot(m200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


df1 = dataa.DataReader(title,'yahoo',START,TODAY)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training = pd.DataFrame(df1['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df1['Close'][0:int(len(df)*0.70): int(len(df))])

data_training_array = scaler.fit_transform(data_training)

model = load_model('stock_pred.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[100-i:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted=model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_predicted = y_predicted.reshape(-1,)
st.subheader('Prediction')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_predicted,'r',label='Predicted Price')
st.pyplot(fig2)







