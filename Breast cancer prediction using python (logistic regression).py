#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


data = pd.read_csv("data.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


# visualize NAs in heatmap
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[6]:


# drop id and empty column
data.drop(['Unnamed: 32', "id"], axis=1, inplace=True)


# In[7]:


# turn target variable into 1s and 0s
data.diagnosis =[1 if value == "M" else 0 for value in data.diagnosis]


# In[8]:


# turn the target variable into categorical data
data['diagnosis'] = data['diagnosis'].astype('category',copy=False)
plot = data['diagnosis'].value_counts().plot(kind='bar', title="Class distributions \n(0: Benign | 1: Malignant)")
fig = plot.get_figure()


# In[9]:


# Prepare the model
y = data["diagnosis"] # our target variable
X = data.drop(["diagnosis"], axis=1) # our predictors


# In[10]:


from sklearn.preprocessing import StandardScaler

# Creating a scaler object
scaler = StandardScaler()

# Fitting the scaler to the data and transforming the data
X_scaled = scaler.fit_transform(X)

# X_scaled is now a numpy array with normalized data


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)


# In[12]:


from sklearn.linear_model import LogisticRegression

# Create logistic regression model
lr = LogisticRegression()

# Train the model on the training data
lr.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred = lr.predict(X_test)


# In[13]:


from sklearn.metrics import accuracy_score

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[14]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:


#We used data from the open dataset Breast Cancer to construct a model that will predict if a given cell is malignant or not based on certain measurements of its nucleus.

