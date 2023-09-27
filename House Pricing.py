#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


df=pd.read_csv("C:/Windows/Lenovo/ImController/Data/House_Price.csv",header=0)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


sns.jointplot(x='n_hot_rooms',y='price',data=df)


# In[8]:


sns.jointplot(x='rainfall',y='price',data=df)


# In[9]:


df.head()


# In[10]:


sns.countplot(x='airport',data=df)


# In[11]:


sns.countplot(x='waterbody',data=df)


# In[12]:


sns.countplot(x='bus_ter',data=df)


# 1.Missing values in the n_hot_rooms
# 2.skewness or outliers in crime rate
# 3.outliers in n_hot_rooms and rainfall
# 4.bus_ter takes only'YES' values

# In[13]:


df.info()


# Treating outliers

# In[14]:


np.percentile(df.n_hot_rooms,[99])


# In[15]:


np.percentile(df.n_hot_rooms,[99])[0]


# In[16]:


uv=np.percentile(df.n_hot_rooms,[99])[0]


# In[17]:


df[(df.n_hot_rooms>uv)]


# In[18]:


df.n_hot_rooms[(df.n_hot_rooms>3*uv)]=3*uv


# In[19]:


np.percentile(df.rainfall,[1][0])


# In[20]:


lv=np.percentile(df.rainfall,[1][0])


# In[21]:


df[(df.rainfall<lv)]


# In[22]:


df.rainfall[(df.rainfall<0.3*lv)]=0.3*lv


# In[23]:


sns.jointplot(x='crime_rate',y='price',data=df)


# In[24]:


df.describe()


# Treating missing values

# In[25]:


df.info()


# In[26]:


df.n_hos_beds=df.n_hos_beds.fillna(df.n_hos_beds.mean())


# In[27]:


df.info()


# In[28]:


df=df.fillna(df.mean()) #for all missing values in data


# In[29]:


sns.jointplot(x='crime_rate',y='price',data=df)


# In[30]:


df.crime_rate=np.log(1+df.crime_rate)


# In[31]:


sns.jointplot(x='crime_rate',y='price',data=df)


# In[32]:


df['avg_dist']=(df.dist1+df.dist2+df.dist3+df.dist4)/4 #adding varible avg_dist


# In[33]:


df.describe()


# In[34]:


del df['dist1']


# In[35]:


df.describe()


# In[36]:


del df['dist2']


# In[37]:


del df['dist3']


# In[38]:


del df['dist4']


# In[39]:


df.describe()


# In[40]:


del df['bus_ter']


# In[41]:


df.head()


# Dummy variable creation for categorical variables

# In[42]:


df=pd.get_dummies(df)


# In[43]:


df.head()


# In[44]:


del df['airport_NO']


# In[45]:


del df['waterbody_None']


# In[46]:


df.head()


# Corelation analysis 

# In[47]:


df.corr() #correlation matrix
#multicorreality


# In[48]:


del df['parks']


# In[49]:


df.head()


# Regression model

# In[50]:


import statsmodels.api as sn


# In[51]:


x=sn.add_constant(df['room_num'])


# In[52]:


lm=sn.OLS(df['price'],x).fit()


# In[53]:


lm.summary()


# In[54]:


from sklearn.linear_model import LinearRegression


# In[55]:


y=df['price']


# In[56]:


X=df[['room_num']]


# In[57]:


lm2=LinearRegression()


# In[58]:


lm2.fit(X,y)


# In[59]:


print(lm2.intercept_,lm2.coef_)


# In[60]:


lm2.predict(X)


# In[61]:


help(sns.jointplot)


# In[62]:


sns.jointplot(x=df['room_num'],y=df['price'],data=df,kind='reg')


# Multiple regression model

# In[63]:


x_multi=df.drop('price',axis=1) #independet variable


# In[64]:


x_multi.head()


# In[65]:


y_multi=df['price']


# In[66]:


y_multi.head()


# In[67]:


x_multi_cons=sn.add_constant(x_multi)


# In[68]:


x_multi_cons.head()


# In[69]:


lm_multi=sn.OLS(y_multi,x_multi_cons).fit()


# In[70]:


lm_multi.summary()


# In[71]:


lm3=LinearRegression()


# In[72]:


lm3.fit(x_multi,y_multi)


# In[73]:


print(lm3.intercept_,lm3.coef_)


# Test,Train ,Split data

# In[75]:


from sklearn.model_selection import train_test_split


# In[77]:


x_train,x_test,y_train,y_test =train_test_split(x_multi,y_multi,test_size=0.2,random_state=0)


# In[79]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[80]:


lm_a=LinearRegression()


# In[81]:


lm_a.fit(x_train,y_train)


# In[82]:


y_test_a=lm_a.predict(x_test)


# In[83]:


y_train_a=lm_a.predict(x_train)


# In[86]:


from sklearn.metrics import r2_score


# In[87]:


r2_score(y_test,y_test_a)


# In[88]:


r2_score(y_train,y_train_a)


# In[ ]:




