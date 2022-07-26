# Import All dependancies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow import keras
import streamlit as st
from datetime import datetime


#Extract file from the source and read it in dataframe

df = pd.read_csv('https://raw.githubusercontent.com/AntonyDsouza1306/Candy_Time_Series_Forecasting/main/candy_production.csv',index_col='observation_date',parse_dates=True)
df.index.freq='MS'

# Clean the data
df.rename(columns = {'IPG3113N':'Per Capita Production'}, inplace = True)

df = df[~(df.index.year == 2017)]

# Slice last 5 five record and kept in another dataframe
df1 = df[df.index.year > 2011]

st.header('Time Series Forecasting for the US Candy Per Capita Production')

# Plot the line chart in streamlit
st.line_chart(df)

# Prepare data
df['year'] = [d.year for d in df.index]
df['month'] = [d.strftime('%b') for d in df.index]
years = df['year'].unique()


#Box plot Year Wise and seasonality Wise in streamlit
#fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
#sns.boxplot(x='year', y='Per Capita Production', data=df, ax=axes[0])
#sns.boxplot(x='month', y='Per Capita Production', data=df.loc[~df.year.isin([1972, 2016]), :])

# Set Title
#axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
#axes[0].tick_params(axis='x', rotation=90)
#axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
#plt.show()
#st.pyplot(fig)


#train and test split
train1 = df.iloc[:528, 0]
test1 = df.iloc[528:, 0]

train = np.asarray(train1)
train = np.reshape(train, (528, 1))
test = np.asarray(test1)
test = np.reshape(test, (12, 1))

# Normalization
scaler = MinMaxScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# Time Series Generator
n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# It can be used to reconstruct the model identically.
model = keras.models.load_model("my_model")

st.header('Select month and Year to Forecast the Candy Per Capita Production ')
# Create a date slider
cols1,_ = st.columns((1,2))
format = 'MM, YYYY'
start_date = '01/2017'
start_date = datetime.strptime(start_date,'%m/%Y')
print(start_date)
end_date = '12/2027'
end_date = datetime.strptime(end_date,'%m/%Y')
print(end_date)
present_date = '06/2022'
present_date = datetime.strptime(present_date,'%m/%Y')
print(present_date)

slider = cols1.slider('Select date', min_value=start_date, value=present_date ,max_value=end_date, format=format)

year_length = (slider.year-2016)*12

# forecast for the future

test_predictions = []

first_eval_batch = scaled_test
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(year_length):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
true_predictions = true_predictions.flatten()


rng = pd.date_range('2017-01-01', periods = year_length, freq='M')
df2 = pd.DataFrame({ 'Per Capita Production' : true_predictions }, index=rng)

df3 = pd.concat([df1, df2], axis=0)

df4 = df3
df4['year'] = df4.index.year
df4 = df4.groupby('year').sum()

df5 = df2.head(year_length-12+slider.month)
df6 = pd.concat([df1, df5], axis=0)
df6['month'] = df6.index.month_name()


fig = plt.figure()
plt.plot(df1.index,df1['Per Capita Production'])
plt.plot(df5.index,df5['Per Capita Production'])
plt.xlabel('Year')
plt.ylabel('Per Capital Production')
plt.title('Forecasting Candy Per Capita Production')
plt.grid(True)
plt.show()
st.pyplot(fig)

st.write(round(df5['Per Capita Production'].iloc[-1],2),' is the US Candy Per Capita Production for ',slider.strftime('%B'),' ',slider.year)



fig = plt.figure(figsize=(10,5))
plt.bar(df6.tail(6).index.month_name(), df6.tail(6)['Per Capita Production'],color ='maroon',width = 0.4)
plt.title('Per Capita Production forecasting for the last 6 months')
plt.ylabel('Per Capita Production', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.show()
st.pyplot(fig)

fig = plt.figure(figsize=(10,5))
plt.bar(df4.tail(5).index, df4.tail(5)['Per Capita Production'],color ='green',width = 0.4)
plt.title('Per Capita Production forecasting for the last 5 years')
plt.ylabel('Per Capita Production', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.show()
st.pyplot(fig)


