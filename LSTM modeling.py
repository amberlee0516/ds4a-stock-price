#!/usr/bin/env python
# coding: utf-8

# # LSTM

# ### Set up

# In[3]:


# import sys

# !{sys.executable} -m pip install tensorflow-macos
# !{sys.executable} -m pip install tensorflow-metal
# !{sys.executable} -m pip install keras

# # if using M1 chip in apple, make sure to update anaconda


# In[4]:


import pandas as pd
from numpy import array
import datetime as dt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# ### About LSTM
# 
# [Tutorial](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)

# ### Dataset
# 
# * 01/01/2018 - 07/01/2023
# * Train: 2018-2021
# * Test: 2022-July 2023 (tech recession!)
# * Companies: Amazon, Apple, Google, Microsoft, Nvidia

# In[5]:


# # to get this file, first run the notebook: Retrieve entire stock price data.ipynb
# stocks = pd.read_csv('quandl_data_table_downloads/QUOTEMEDIA/PRICES_20230712.zip')

# company_tickers = ['AMZN', 'AAPL', 'GOOG', 'MSFT', 'NVDA']
# start_date = pd.to_datetime('2017-12-01')
# end_date = pd.to_datetime('2023-07-01')

# stocks = stocks.loc[stocks['ticker'].isin(company_tickers)]
# stocks = stocks[['date', 'ticker', 'adj_close']]
# stocks['date'] = pd.to_datetime(stocks['date'])
# stocks = stocks.loc[(stocks['date'] >= pd.to_datetime(start_date))
#                       & (stocks['date'] <= pd.to_datetime(end_date))]
# stocks = stocks.sort_values('date')

# stocks.to_csv('stocks_filtered.csv', index=False)

# # # this will be needed later to merge with sentiment analysis dataset
# # stocks = stocks.set_index('date').tz_localize('utc')


# In[6]:


# stocks.shape # there are missing dates


# In[7]:


# 1 - (7015/(365*5))  / 5.5 # percent of missing dates


# ### Directly read data

# In[8]:


stocks = pd.read_csv('stocks_filtered.csv')
stocks['date'] = pd.to_datetime(stocks['date'])


# In[9]:


stocks.head()


# ### Baseline LSTM
# 
# No sentiment analysis; only one company

# In[10]:


nstocks = stocks.loc[stocks['ticker'] == "NVDA"]
nstocks = nstocks.sort_values('date').set_index('date')
nstocks.head()


# In[11]:


def split_data(df, n_steps, count_imputations=False,
               start_date='2018-01-01', end_date='2023-07-01'):
    """
    reformats stock price data to be a sequence of prices from n_steps days ago
    fills in missing values with the most recent available price data
    
    has an option to count the number of imputations
    
    returns three arrays:
    1. y, 
    2. X, each element has length n_steps 
    3. imputations as arrays, where X has n_steps number of columns of 
    previous n_steps stock prices
    
    df: DataFrame with 'date' and 'adj_close' price columns
    n_steps: look back window. < 30
    
    left and right inclusive
    """
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    all_dates = pd.date_range(start = start_dt, end = end_dt)
    missing_dates = all_dates.difference(df.index)
    y_dates = df.index[(df.index >= start_dt) &
                       (df.index <= end_dt)]
    
    delta = pd.Timedelta(str(n_steps) + " days")
    
    y = []
    X = []
    imputations = []

    for y_date in y_dates:

        y.append(df.adj_close.loc[y_date])

        # dates with price data
        X_dates = nstocks.index[(df.index >= y_date - delta) & 
                                (df.index < y_date)]

        all_X_dates = pd.date_range(start = y_date - delta, 
                                    end = y_date,
                                    inclusive = "left") # exclude y_date

        missing = all_X_dates.difference(X_dates)

        X_prices = []
        count_imputations = 0

        for date in all_X_dates:

            if date in missing:

                # most recent date with price data
                impute_date = max(df.index[df.index < date])
                X_prices.append(df['adj_close'].loc[impute_date])

                count_imputations += 1

            else:

                X_prices.append(df['adj_close'].loc[date])

        X.append(X_prices)
        imputations.append(count_imputations)
        
    if count_imputations:
        return array(X), array(y), array(imputations)
    
    return array(X), array(y)
    

    


# In[12]:


# X, y, imputations = split_data(nstocks, 5, True)


# In[13]:


# pd.DataFrame({"imputations": imputations}).hist()


# ### Train test split

# In[14]:


train = split_data(nstocks, 5, start_date='2018-01-01', end_date='2021-12-31')
X_train, y_train = train[0], train[1]


# In[15]:


test = split_data(nstocks, 5, start_date='2022-01-01', end_date='2023-07-01')
X_test, y_test = test[0], test[1]


# In[16]:


n_steps = 5
n_features = 1 # univariate time series


# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# switching relu activiation to tanh


# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))


# In[17]:


X_train.shape


# In[ ]:


model.fit(X_train, y_train, epochs=20, verbose=0)


# In[7]:


#import tensorflow as tf
#tf.config.list_physical_devices()


# In[ ]:





# In[ ]:


#with tf.device('/device:GPU:0'):
#    model.fit(X_train, y_train, epochs=20, verbose=0)

