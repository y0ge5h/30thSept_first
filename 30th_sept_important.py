
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split as tts 
from sklearn.metrics import mean_squared_error , r2_score

df = pd.read_csv("D:/grey work/data_sets/train.csv")


# In[84]:


df.head()


# In[85]:


correlation_matrix = df.select_dtypes(include=[np.number]).corr()


# In[86]:


correlation_matrix


# In[87]:


selected_features = correlation_matrix[["SalePrice"]][(correlation_matrix["SalePrice"]>=0.6)|(correlation_matrix["SalePrice"]<=-0.6)]


# In[88]:


selected_features


# In[100]:


X = df[['OverallQual','TotalBsmtSF','GrLivArea', 'GarageArea']]
y = df['SalePrice']

X_train , X_test , y_train , y_test = tts(X ,y ,test_size=0.3 , random_state = 42 )


# In[101]:


reg = LinearRegression()


# In[102]:


reg.fit(X_train,y_train)


# In[103]:


y_pred = reg.predict(X_test)


# In[104]:


reg.score(X_test,y_test)


# In[105]:


mse = mean_squared_error(y_pred,y_test)


# In[106]:


rmse = mse**0.5


# In[107]:


rmse


# In[108]:


r_score = r2_score(y_pred,y_test)


# In[109]:


r_score

