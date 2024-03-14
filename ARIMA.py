#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy openpyxl')


# In[2]:


import pandas as pd
import numpy as np
from datetime import timedelta

# Define the path to your Excel file
excel_path = r'Worksheet.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_path)


# In[3]:


df


# In[4]:


# Ensure the 'Date' column is of datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by the 'Date' column
df.sort_values(by='Date', inplace=True)

# Generate a list of consecutive dates
start_date = df['Date'].min()
end_date = df['Date'].max()


# In[5]:


date_range = pd.date_range(start_date, end_date, freq='D')


# In[6]:


date_range


# In[7]:


# Find missing dates
missing_dates = date_range[~date_range.isin(df['Date'])]

# Create a DataFrame for missing dates with NaN values for 'Price'
missing_df = pd.DataFrame({'Date': missing_dates, 'Price': np.nan})


# In[8]:


# Concatenate the original DataFrame and the missing dates DataFrame
df = pd.concat([df, missing_df], ignore_index=True)

# Sort the DataFrame by 'Date' again
df.sort_values(by='Date', inplace=True)


# In[9]:


# Fill missing 'Price' values with the mean of the existing 'Price' values
df['Price'].fillna(df['Price'].mean(), inplace=True)


# In[10]:


# Save the updated DataFrame to a new Excel file
output_excel_path = r'C:\Users\HP\Downloads\Time Series\wheat_price_viram_updated.xlsx'
df.to_excel(output_excel_path, index=False)


# In[11]:


print(f"Missing dates found: {len(missing_dates)}")


# In[12]:


# Define the path to your Excel file
excel_path = r'C:\Users\HP\Downloads\Time Series\wheat_price_viram_updated.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_path)


# In[13]:


df


# In[15]:


# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(df['Price'])
plt.title('Price analysis')
plt.grid(True)
plt.show();


# In[16]:


df.describe()


# In[17]:


from statsmodels.tsa.stattools import adfuller
test_result=adfuller(df['Price'])
test_result


# In[18]:


import matplotlib.pyplot as plt
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Price'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Price'], lags=40, ax=ax2)
plt.show()


# In[19]:


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
#Plots with first order differenced Sales variable
fig = sm.graphics.tsa.plot_acf(df['Price'].diff().dropna(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Price'].diff().dropna(), lags=40, ax=ax2)
plt.show()


# In[25]:


import pmdarima as pm
import matplotlib.pyplot as plt


# In[20]:


excel_path = r'C:\Users\HP\Downloads\Time Series\wheat_price_viram_updated.xlsx'


# In[21]:


df = pd.read_excel(excel_path)


# In[22]:


df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by the 'Date' column
df.sort_values(by='Date', inplace=True)

# Check for missing values in 'Price' column and fill them with the mean
if df['Price'].isna().any():
    df['Price'].fillna(df['Price'].mean(), inplace=True)


# In[23]:


# Normalize or scale the 'Price' column as needed (e.g., using Min-Max scaling)
df['Price'] = (df['Price'] - df['Price'].min()) / (df['Price'].max() - df['Price'].min())

# Split the data into training and testing sets (e.g., 80% train, 20% test)
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]


# In[26]:


# Fit an AutoARIMA model to the training data
model = pm.auto_arima(train_data['Price'], seasonal=True, m=7, trace=True)

# Print the model summary
print(model.summary())


# In[27]:


# Make predictions on the training data
train_predictions, _ = model.predict_in_sample(return_conf_int=True)

# Store the training predictions with dates
train_predictions_df = pd.DataFrame({'Date': train_data['Date'], 'Predicted_Price': train_predictions})
train_predictions_df.set_index('Date', inplace=True)

# Make predictions on the test data
test_predictions, conf_int = model.predict(n_periods=len(test_data), return_conf_int=True)

# Store the test predictions with dates
test_predictions_df = pd.DataFrame({'Date': test_data['Date'], 'Predicted_Price': test_predictions})
test_predictions_df.set_index('Date', inplace=True)

# Forecast 10 steps ahead
forecast_steps = 60
forecast, conf_int_forecast = model.predict(n_periods=forecast_steps, return_conf_int=True)

# Create a DataFrame for the forecasted values with future dates
future_dates = pd.date_range(df['Date'].max() + timedelta(days=1), periods=forecast_steps, freq='D')
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Price': forecast})
forecast_df.set_index('Date', inplace=True)


# In[28]:


# Plot actual series (train and test), predicted series (train, test, and forecast)
plt.figure(figsize=(12, 6))
plt.plot(train_data['Date'], train_data['Price'], label='Train Data', color='blue')
plt.plot(test_data['Date'], test_data['Price'], label='Test Data', color='green')
plt.plot(train_predictions_df.index, train_predictions_df['Predicted_Price'], label='Train Predictions', color='purple')
plt.plot(test_predictions_df.index, test_predictions_df['Predicted_Price'], label='Test Predictions', color='orange')
plt.plot(forecast_df.index, forecast_df['Forecasted_Price'], label='10-Step Ahead Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('Time Series Forecasting')
plt.grid(True)
plt.show()


# In[ ]:




