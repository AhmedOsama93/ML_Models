#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
from statistics import mode
get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[67]:


df = pd.read_csv("BankNote_Authentication.csv")


# In[68]:


df.head()


# In[69]:


df.describe()


# In[70]:


#shuffle our data
shuffled_df = normalized_df.sample(frac=1).reset_index(drop=True)
shuffled_df.head()


# In[71]:


data_len = len(shuffled_df.index)


# In[72]:


split_at = int(data_len*0.3)
test_data, train_data = shuffled_df[:split_at], shuffled_df[split_at:]


# In[73]:


test_data = test_data.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)


# In[74]:


len(test_data.index)


# In[75]:


len(train_data.index)


# In[76]:


unnormalized_trin_class=train_data['class']
unnormalized_test_class=test_data['class']

mean = train_data.mean()
std = train_data.std()

normalized_train_df = (train_data-mean)/std
normalized_test_df = (test_data-mean)/std


normalized_train_df['class']=unnormalized_trin_class
normalized_test_df['class']=unnormalized_test_class





# In[77]:


normalized_train_df.head()


# In[78]:


normalized_test_df.head()


# In[79]:


def mean_squared_error(data,point):
    m,n = data.shape
    error=[]
    for i in range(m):
        row_error=0
        for j in range(n):
            row_error+= pow(data.loc[i][j] - point[j],2)
        error.append(row_error)
    return error


# In[80]:


def knn(features,label,features_to_predict,k):
    error = mean_squared_error(features,features_to_predict)
    df = features.copy()
    df['label']=label
    df['error']=error
    df = df.sort_values(by=['error'])
    df_top_k = df.head(k)
    return df_top_k['label'].mode()[0],list(df['label'])


# In[ ]:


n,m = test_data.shape
correct_predections_list=[]
train_features = normalized_train_df.iloc[:,:-1]
train_label = normalized_train_df['class'].tolist()

test_features = normalized_test_df.iloc[:,:-1]
test_label = normalized_test_df['class']

k_values_list = []
for j in range(n):
        predected_value,all_error_sorted = knn(train_features,train_label,test_features.iloc[j],1)
        k_values_list.append(all_error_sorted)


# In[ ]:


for i in range (9):
    correct_predections=0
    for j in range(n):
        predected_value = mode(k_values_list[j][0:i+1])
        if(predected_value == test_label[j]):
            correct_predections+=1
    print('k value : ',i+1)
    print('Number of correctly classified instances : ',correct_predections,'Total number of instances : ',len(test_data.index))
    print('Accuracy : ',correct_predections/len(test_data.index)*100)
    correct_predections_list.append(correct_predections)


# In[ ]:


correct_predections_list


# In[ ]:





# In[ ]:





# In[ ]:




