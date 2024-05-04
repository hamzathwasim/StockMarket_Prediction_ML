#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('/Users/wasim/Desktop/WASIM/Projects/stock/stocks.csv')
data


# In[2]:


data.isnull().sum()


# In[3]:


data.columns


# In[4]:


data.shape


# In[5]:


data.head(10)


# In[6]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
data[["Volume"]]=sc.fit_transform(data[["Volume"]])
data


# In[7]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close',
       'Volume']
for i in data1:
    data[i]=le.fit_transform(data[i])
data.head(10)


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
cor=data.corr()
sns.heatmap(cor,annot=True,cmap="viridis")
plt.show()


# In[9]:


x=data.drop('Close',axis=1)
x.head(10)


# In[10]:


y=data['Close']
y.head(10)


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[12]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.29) 
regr=LinearRegression()


# In[13]:


regr.fit(xtrain,ytrain)


# In[14]:


from sklearn.metrics import accuracy_score


# In[15]:


accuracy=(regr.score(xtest,ytest)*100)
accuracy


# In[16]:


predict=(regr.score(xtrain,ytrain)*100)
predict


# In[17]:


v=[[0,2,84,81,77,71,211]]
w=regr.predict(v)
w


# In[18]:


ypredict=regr.predict(xtest)
ypredict


# In[19]:


ytest


# In[20]:


w1=ypredict.reshape(len(ypredict),1)
w1


# In[21]:


w2=ytest.values.reshape(len(ytest),1)
w2


# In[22]:


from sklearn.metrics import r2_score


# In[23]:


r2_score(ytest,ypredict)


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


x=data["Open"]
y=data["Close"]
plt.scatter(x,y)
plt.xlabel("Open")
plt.ylabel("Close")
plt.show()


# In[ ]:




