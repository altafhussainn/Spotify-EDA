#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Importing the datasets downloaded from kaggle.


# In[4]:


df1 = pd.read_csv('tracks.csv')
df1.head()


# In[ ]:


# Null values 


# In[5]:


pd.isnull(df1).sum()


# In[ ]:


# Exploring the data


# In[6]:


df1.info()


# In[7]:


# least popular


# In[10]:


sorted_df = df1.sort_values('popularity', ascending = True).head(10)
sorted_df


# In[12]:


# descriptive satistics


# In[11]:


df1.describe().transpose()


# In[13]:


# most popular


# In[14]:


most_popular = df1.query('popularity>90', inplace = False).sort_values('popularity', ascending = False)
most_popular[:10]


# In[23]:


df1.head()


# In[22]:


df1.head()


# In[24]:


df1[['artists']].iloc[18]


# In[25]:


df1[['artists']].iloc[322]


# In[26]:


# converting the duration from ms to s


# In[28]:


df1['duration'] = df1['duration_ms'].apply(lambda x: round(x/1000))
df1.drop('duration_ms', inplace = True, axis = 1)
df1.head()


# In[29]:


df1.duration.head()


# In[38]:


numeric_cols = df1.select_dtypes(include=['float64', 'int64']).columns
corr_df = df1[numeric_cols].corr(method="pearson")
plt.figure(figsize=(14, 6))
heatmap = sns.heatmap(corr_df, annot=True, fmt=" .1g", vmin=-1, vmax=1, center=0, cmap="inferno", linewidths=1, linecolor='black')
heatmap.set_title("Correlation HeatMap Between Variables")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
plt.show()


# In[40]:


sample_df = df1.sample(int(0.004*len(df1)))
print(len(sample_df))


# In[41]:


plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y = 'loudness', x = 'energy', color = 'c').set(title ="Loudness Vs Energy Correlation" )


# In[43]:


plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y = 'popularity', x = 'acousticness', color = 'b').set(title ="Popularity Vs Acousticness Correlation" )


# In[ ]:





# In[ ]:





# In[ ]:




