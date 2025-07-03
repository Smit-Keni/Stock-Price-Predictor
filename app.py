import pandas as pd
import numpy as np
import pandas_datareader as data
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta



st.header("Stock Price Predictor")
end = st.date_input("Enter end date",max_value=dt.date.today())




start = st.date_input("Enter start date",max_value=end)



stock= st.text_input("Enter Stock Symbol","GOOG")

df = yf.download(stock,start,end,auto_adjust=False)

df.to_csv("stock_data.csv")
df = pd.read_csv("stock_data.csv")

df.rename(columns={"Price":"Date"},inplace=True)
df.drop(index=[0,1],inplace=True)
df=df.drop(["Adj Close","Date"],axis=1)
df = df.reset_index(drop=True)
#I had to apply this step for a certain reason
df = df.apply(pd.to_numeric, errors='coerce')

#Describe the data
st.subheader("Data from "+ str(start) +" to " + str(end))
st.write(df.describe())

#graphs

st.subheader("Closing price vs Time")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing price vs Time with MA100")
ma100 = df.Close.rolling(100).mean()

fig = plt.figure(figsize=(12,6))
plt.plot(ma100,"red")
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing price vs Time with MA200")
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200,"green")
plt.plot(ma100,"red")
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
data_training_array = data_training_array.flatten()
len(data_training_array)



model = load_model("stock_model.keras")

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)
input_data = input_data.flatten()

X_test = []
y_test=[]
X_fut_test=X_test
y_fut_test=y_test

for i in range(100,len(input_data)):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i])

X_test,y_test = np.array(X_test),np.array(y_test)

y_predicted = model.predict(X_test)

st.subheader("Predicted(Red) vs Actual(Blue) values")

fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test)
plt.plot(y_predicted,"red")

st.pyplot(fig2)


