#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("nutrient.csv",index_col=0)


# In[3]:


df.head()


# In[4]:


df.shape


# In[7]:


from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
nutrient_std = scaler.fit_transform(df)
nutrient_std = pd.DataFrame(nutrient_std,columns=df.columns,index=df.index)

nutrient_std.head()


# In[8]:


from sklearn.cluster import KMeans

kscore=[2,3,4,5,6,7,8,9,10]
inertia=[]

for i in kscore :
    model = KMeans(n_clusters=i,random_state=2020)
    model.fit(nutrient_std)
    inertia.append(model.inertia_)


# In[9]:


inertia


# ## Elbow Plot
# picking the elbow of the curve as the number of clusters to use.

# In[10]:


#pyplot:
    
import matplotlib.pyplot as plt

plt.plot(kscore,inertia,'-o')
plt.title('score plot ')
plt.xlabel('kscore, k ')
plt.ylabel('inertia')
plt.xticks(kscore)
plt.show()


# In[11]:


# #kmeans instance with best k

model=KMeans(n_clusters=5)

# #fitting the model:
model.fit(nutrient_std)

# #labels:
labels = model.predict(nutrient_std)


# ## Data Prediction

# In[12]:


labels


# In[13]:


clusterID = pd.DataFrame({'ClustID':labels},index=df.index)
clusteredData= pd.concat([df,clusterID],axis='columns')


# In[14]:


clusterID.head()


# In[15]:


clusteredData


# In[16]:


clf=clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')

print(clf)


# ## Checking Outliers

# In[ ]:


df_outlier = pd.read_csv("nutrient.csv")


# In[ ]:


df_outlier.columns


# In[ ]:


df_outlier.describe()


# In[ ]:


# from above we can see that for column calcium the mean is very low compare to its max value
# hence there are chances of outliers

# THIS WE OBSERVED THROUGH SIMPLE MDESCRIBE FUNCTION


# In[ ]:


# LETS LOOK THROUGH VISUALIZATION


# In[ ]:


## LOOKING FOR CALCIUM ATTRIBUTE

import plotly.express as px
fig = px.scatter(df_outlier,x = "calcium")
fig.show()


# In[ ]:


# LOOKING FOR PROTEIN ATTRIBUTE

fig = px.scatter(df_outlier, x="protein")
fig.show()


# In[ ]:


fig = px.box(df_outlier, x='energy')
fig.show()


# In[ ]:


fig = px.histogram(df_outlier, x='energy')

fig.show()


# In[ ]:




