#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[3]:


df1 = pd.read_csv("car_data.csv")


# In[4]:


df1.head()


# In[5]:


df1.describe()


# In[6]:


numeric_variables = df1[['ID','symboling','wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']].copy()


# In[7]:


for i in range(len(numeric_variables.columns)-1):
    f = plt.scatter(numeric_variables["price"],numeric_variables.iloc[:,i])
    plt.xlabel("Price")
    plt.ylabel(numeric_variables.columns[i])
    plt.show()
    print("\n")


# ### the most 4 correlated features with the price are:
# >carwidth
# 
# >curbweight
# 
# >enginesize
# 
# >horsepower

# In[8]:


new_df = df1[['carwidth','curbweight','enginesize','horsepower','price']].copy()
new_df.head()


# In[9]:


min_value = new_df.min()
max_value = new_df.max()
max_value


# In[10]:


normalized_df = (new_df - min_value) / (max_value - min_value)
normalized_df


# In[177]:


#shuffle our data
shuffled_df = normalized_df.sample(frac=1).reset_index(drop=True)
shuffled_df.head()


# In[178]:


len(shuffled_df.index)


# In[179]:


test_data, train_data = shuffled_df[:40], shuffled_df[40:]


# In[180]:


test_data = test_data.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)


# In[181]:


len(test_data.index)


# In[182]:


len(train_data.index)


# In[212]:


def mean_squared_error(data_x,data_y,slop,y_intercept):
    cost = 0.0
    for i in range(data_x.shape[0]):
        model_value = np.dot(data_x[i] , slop) + y_intercept
        cost = cost + (model_value - data_y[i])**2
    cost = cost/ (2*data_x.shape[0])
    return cost


#  gradint descent :
#  $$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
# \;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w}   \; \newline 
#  b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
# \end{align*}$$
# 
# and the partial derivative can be calculated as :
# $$
# \begin{align}
# \frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \\
#   \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \\
# \end{align}
# $$

# In[213]:


def get_derivative (data_x,data_y,slop_list):
    m,n = data_x.shape
   
    fn_predictions = np.dot(data_x,slop_list)
    diff = fn_predictions - data_y
    error = np.dot(data_x.transpose(),diff)
    error/=m
    return error


# In[214]:


def gradient_descent (data_x,data_y,slop,learning_rate,max_num_iters,threshold):
    cost_every_iteration = []
    for i in range(num_iters):
        descent = learning_rate *get_derivative(data_x,data_y,slop)
        slop-=descent
        cost_every_iteration.append(mean_squared_error(data_x,data_y,slop,y_intercept))
        if(i!=0):
            if(cost_every_iteration[i-1]-cost_every_iteration[i] < threshold):
                break;
    return slop, cost_every_iteration
    


# In[184]:


train_data['ones']=1


# In[191]:


train_data.head()


# In[215]:


x = train_data.drop(columns = ['price']).to_numpy().reshape((-1,5))
y = train_data['price']
slop = [1,1,1,1,1]
y_intercept = 0
learning_rate = 0.1
max_num_iters = 50
threshold = 0.0001
slop, cost_every_iteration = gradient_descent(x,y,slop,learning_rate,max_num_iters,threshold)


# In[216]:


mean_squared_error(x,y,slop, y_intercept)


# In[217]:


plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost against the number of iterations')
_ = plt.plot(cost_every_iteration)


# In[ ]:




