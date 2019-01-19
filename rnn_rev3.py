# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:35:02 2018

@author: Rachid
"""
###............... Collect data ..............###
from pandas_datareader import data
import pandas_datareader.data as web
import datetime
from bokeh.plotting import figure, show, output_file

start=datetime.datetime(2013,11,15)
end = datetime.datetime(2018,11,15)
df = data.DataReader(name="GOOG", data_source='iex', start = start, end = end)

#df.index = df.index.map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))


# Predict the trend for google stock in the future
###................... Build RNN ..............###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train = df

# get an array of the 'open' columns
training_set = dataset_train[['open']].values

# feature scaling (with Normalization)
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creation of neural structure with 60 timestamps and 1 output
X_train = []
y_train = []
for i in range(60,1170):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i, 0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Librairies Keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Initialization
regressor = Sequential()

# LSTM & dropout Layers
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=100, return_sequences=False))
regressor.add(Dropout(0.3))

# Output layer
regressor.add(Dense(units=1))

# Compile
regressor.compile(optimizer='adagrad', loss='mean_squared_error')

# Training
regressor.fit(X_train, y_train, epochs=500, batch_size=16)



###............... Prediction .............###
# Predictions
inputs = df[60:].values.reshape(-1,1)

# feature scaling of inputs (transform, because it already trained in the beginning)
inputs = sc.transform(inputs)

# Reshaping
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
    
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Prediction
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


new_rawInputs = dataset_train['open'][len(dataset_train) - 60:].values
new_rawInputs = np.array(new_rawInputs)
new_rawInputs = new_rawInputs.reshape(-1,1)
new_inputs = sc.transform(new_rawInputs)
new_inputs = new_inputs.reshape(60)
new_inputs = new_inputs.tolist()
 
for i in range(0,20):
    test_inputs = np.array(new_inputs)
    test_inputs = np.reshape(test_inputs, (1, 60, 1))
    new_val = regressor.predict(test_inputs)
    new_inputs = new_inputs[1:] + new_val[0].tolist()
 
longTermPredictedStockPrice = new_inputs[(len(new_inputs) - 20):]
longTermPredictedStockPrice = np.array(longTermPredictedStockPrice)
longTermPredictedStockPrice = longTermPredictedStockPrice.reshape(-1,1)
longTermPredictedStockPrice = sc.inverse_transform(longTermPredictedStockPrice)
 
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.plot(longTermPredictedStockPrice, color='green', label = '20 days Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# relative rmse calcul
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))




