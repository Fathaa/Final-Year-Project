import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','AAPL')
yf.pdr_override()
df = pdr.get_data_yahoo(user_input, start='2010-01-01', end='2022-12-31')

#Describing the data
st.subheader('Data from 2010-2022')
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

data_train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler=MinMaxScaler(feature_range=(0,1))
data_train_arr=scaler.fit_transform(data_train)

#Load model
model=load_model('keras_model.h5')

#Testing part
past_100_days=data_train.tail(100)
final_df=past_100_days.append(data_test,ignore_index=True)
input_data=scaler.fit_transform(final_df)

#Splitting data into x_test and y_test
x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_pred=model.predict(x_test)

scale_offset=scaler.scale_
scale_factor=1/scale_offset[0]

y_pred=y_pred*scale_factor
y_test=y_test*scale_factor

#Final Graph
st.subheader('Predicted vs Original')

fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="Original Price")
plt.plot(y_pred,'r',label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


