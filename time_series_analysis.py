#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
#plt.style.use('fivethirtyeight')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import math
from sklearn.metrics import mean_squared_error
from random import random
import datetime


# In[ ]:


# for getting accss of gdrive file
#from google.colab import drive
#drive.mount('/content/contentprice.csv')
#import your dataframe
df = pd.read_excel("Arrival.xlsx")


# In[ ]:


#########file_path = r'C:\Users\ASUS\Downloads\ISRO DATA\Main data\modeldata.xlsx'

######sheet_name = 'dummydata'
#df = pd.read_excel(file_path, sheet_name=sheet_name)
##########


# In[ ]:


df


# In[ ]:


df['Date']=pd.to_datetime(df['Date'],infer_datetime_format=True)
df.head(1)


# In[ ]:


df.set_index('Date', inplace=True)
df.head(1)


# In[ ]:


# Visualize
plt.figure(figsize=(10, 5))
plt.plot(df['Price'])
plt.title('Price analysis')
plt.grid(True)
plt.show();


# ************ARIMA and Seasonal ARIMA
# Autoregressive Integrated Moving Averages
# STEPS:
# Visualize the Time Series Data
# Make the time series data stationary
# Plot the Correlation and AutoCorrelation Charts
# Construct the ARIMA Model or Seasonal ARIMA based on the data
# Use the model to make predictions
# **********************[link text](https://)

# In[ ]:


df.describe()


# In[ ]:


### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller
test_result=adfuller(df['Price'])


# In[ ]:


test_result


# In[ ]:


### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller
test_result1=adfuller(df['Price'].diff().dropna())
test_result1


# #Ho: It is non stationary
# #H1: It is stationary

# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Price'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Price'], lags=40, ax=ax2)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
#Plots with first order differenced Sales variable
fig = sm.graphics.tsa.plot_acf(df['Price'].diff().dropna(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Price'].diff().dropna(), lags=40, ax=ax2)
plt.show()


# In[ ]:



X = df['Price']
X = X.values
train, test = X[0:len(X)-60], X[len(X)-60:]


# In[ ]:





# In[ ]:


get_ipython().system('pip install pmdarima --quiet')


# In[ ]:


from pmdarima.arima import auto_arima


# In[ ]:


arima_model =  auto_arima(train,start_p=0, d=1, start_q=0,
                          max_p=5, max_d=5, max_q=5, start_P=0,
                          D=1, start_Q=0, max_P=5, max_D=5,
                          max_Q=5, m=12, seasonal=True,
                          error_action='warn',trace = True,
                          supress_warnings=True,stepwise = True,
                          random_state=20,n_fits = 50 )


# In[ ]:





# In[ ]:





# In[ ]:


#Summary of the model
arima_model.summary()


# In[ ]:


arima_model.plot_diagnostics(figsize=(15,12))
plt.show()


# In[ ]:


def forecast(ARIMA_model, periods=60):
    # Forecast
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    #pred = ARIMA_model.predict(n_periods=len(df["#y"]),dynamic=True)
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(days=1), periods = n_periods, freq='D')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(df["price"], color='#1f76b4')
    #plt.plot(pred, color='yellow')
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index,
                    lower_series,
                    upper_series,
                    color='k', alpha=.15)

    plt.title("ARIMA - Forecast")
    plt.show()


# In[ ]:


forecast(arima_model)


# In[ ]:


#nonsesonal
#Standard ARIMA Model
#ARIMA_model = auto_arima(df['price'],
ARIMA_model = auto_arima(train,
                      start_p=1,
                      start_q=1,
                      test='adf', # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                      d=None,# let model determine 'd'
                      seasonal=False, # No Seasonality for standard ARIMA
                      trace=True, #logs
                      error_action='warn', #shows errors ('ignore' silences these)
                      suppress_warnings=True,
                      stepwise=True, n_fits = 50 )


# In[ ]:


forecast(ARIMA_model)


# In[ ]:


# SARIMAX Model
#SARIMAX_model = auto_arima(df[['price']], exogenous=df[['arrival', 'CoC', 'CoP']],
#exogenous=df[['arrival', 'CoC', 'CoP'][0:len(df)-60]]
SARIMAX_model = auto_arima(train, exogenous=df[['arrival', 'CoC', 'CoP'][0:len(df)-60]],
                           start_p=0, d=1, start_q=0,
                          max_p=5, max_d=5, max_q=5, start_P=0,
                          D=1, start_Q=0, max_P=5, max_D=5,
                          max_Q=5, m=12, seasonal=True,
                          error_action='warn',trace = True,
                          supress_warnings=True,stepwise = True,
                          random_state=20,n_fits = 10)


# In[ ]:





# In[ ]:


#Summary of the model
SARIMAX_model.summary()


# In[ ]:


SARIMAX_model.plot_diagnostics(figsize=(15,12))
plt.show()


# In[ ]:


forecast(SARIMAX_model)


# In[ ]:


def forecasts(SARIMAX_model, periods=60):
    # Forecast
    n_periods = periods
    fitted, confint = SARIMAX_model.predict(n_periods=n_periods, return_conf_int=True)
    #pred = ARIMA_model.predict(n_periods=len(df["#y"]),dynamic=True)
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(days=1), periods = n_periods, freq='D')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(df["price"], color='#1f76b4')
    #plt.plot(pred, color='yellow')
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index,
                    lower_series,
                    upper_series,
                    color='k', alpha=.15)

    plt.title("SARIMAX - Forecast")
    plt.show()


# In[ ]:


forecasts(SARIMAX_model)


# In[ ]:




