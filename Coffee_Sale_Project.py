#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np


# In[36]:


pip install pandas scikit-learn matplotlib seaborn


# In[48]:


# Load data set
data = pd.read_csv('index.csv')


# In[38]:


print(data.head())


# In[39]:


print(data.tail())


# In[40]:


print(data.isnull().sum())


# In[41]:


data = pd.DataFrame()


# In[42]:


print(pd.DataFrame())


# In[43]:


print(data.columns)


# In[52]:


# Rename 'money' column to 'Sales' for consistency with previous analysis
data.rename(columns={'money': 'Sales'}, inplace=True)


# In[53]:


print(data.head())


# In[54]:


# Check for missing values
print(data.isnull().sum())


# In[55]:


# Fill missing numerical values with the median (for 'Sales' column)
data['Sales'].fillna(data['Sales'].median(), inplace=True)


# In[56]:


data['coffee_name'].fillna(data['coffee_name'].mode()[0], inplace=True)
data['cash_type'].fillna(data['cash_type'].mode()[0], inplace=True)


# In[57]:


# Convert 'date' to datetime format
data['date'] = pd.to_datetime(data['date'])


# In[58]:


print(data.head())


# In[59]:


print(data.tail())


# In[62]:


# Convert the Date column to datetime type
data['Date'] = pd.to_datetime(data['date'])


# In[63]:


# Check if the conversion worked
print(data.dtypes)


# In[65]:


# Removing outliers using IQR method for 'Sales' column
Q1 = data['Sales'].quantile(0.25)  # First quartile (25%)
Q3 = data['Sales'].quantile(0.75)  # Third quartile (75%)
IQR = Q3 - Q1  # Interquartile range


# In[66]:


# Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[67]:


# Filter out outliers in the 'Sales' column
data_filtered = data[(data['Sales'] >= lower_bound) & (data['Sales'] <= upper_bound)]


# In[68]:


# Check the filtered data
print(data_filtered.head())


# In[69]:


print(data.shape)


# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[71]:


# Set plot style
plt.style.use('seaborn')


# In[72]:


# Plot Sales over time
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Sales'], color='blue')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()


# In[73]:


# Grouping by Month
data['Month'] = data['Date'].dt.to_period('M')
monthly_sales = data.groupby('Month')['Sales'].sum()


# In[74]:


# Plot Monthly Sales
plt.figure(figsize=(10,6))
monthly_sales.plot(kind='bar', color='orange')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()


# In[75]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[76]:


# Feature Engineering - Convert Date to Ordinal for Linear Regression
data['Date_ordinal'] = data['Date'].apply(lambda x: x.toordinal())


# In[77]:


# Selecting Features and Target
X = data[['Date_ordinal']]
y = data['Sales']


# In[78]:


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[79]:


# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)


# In[80]:


# Predicting Sales
y_pred = model.predict(X_test)


# In[81]:


# Visualize Actual vs Predicted Sales
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', label='Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[88]:


# Grouping by 'card' to see total purchases
customer_sales = data.groupby('card')['Sales'].sum().reset_index()


# In[89]:


# Print the total sales per customer
print(customer_sales)


# In[90]:


# Display the grouped data
print(customer_sales)


# In[87]:


print(data.columns)


# In[93]:


# Grouping by CustomerID to see their total purchases
customer_sales = data.groupby('card')['Sales'].sum()


# In[94]:


# Top 5 Customers by Sales
top_customers = customer_sales.sort_values(ascending=False).head(5)
print(top_customers)


# In[95]:


# Visualize the top 5 customers
plt.figure(figsize=(8,4))
top_customers.plot(kind='bar', color='green')
plt.title('Top 5 Customers by Sales')
plt.xlabel('CustomerID')
plt.ylabel('Total Sales')
plt.show()


# In[ ]:




