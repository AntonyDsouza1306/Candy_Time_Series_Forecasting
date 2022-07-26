import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import streamlit as st
from datetime import datetime


#Extract file from the source and read it in dataframe

df = pd.read_csv('https://raw.githubusercontent.com/AntonyDsouza1306/Candy_Time_Series_Forecasting/main/candy_production.csv',index_col='observation_date',parse_dates=True)
df.index.freq='MS'

# Clean the data
df.rename(columns = {'IPG3113N':'Per Capita Production'}, inplace = True)

df = df[~(df.index.year == 2017)]

# Slice last 5 five record and kept in another dataframe
#df1 = df[df.index.year > 2011]

#train and test split
train1 = df.iloc[:516, 0]
test1 = df.iloc[516:, 0]

train = np.asarray(train1)
train = np.reshape(train, (516, 1))
test = np.asarray(test1)
test = np.reshape(test, (24, 1))

# Normalization
scaler = MinMaxScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# Time Series Generator
n_input = 24
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(generator,epochs=50)

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")
# Saving the model
#import pickle
#pickle.dump(model, open('model.pkl', 'wb'))