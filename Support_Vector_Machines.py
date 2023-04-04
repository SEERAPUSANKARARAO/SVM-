#!/usr/bin/env python
# coding: utf-8

# We will work on the implementation of SVM 
# Data: Prediction of the grant of personal loan to the customer based on his demographic and financial attributes
# 
# While solving the continuous value prediction problem, perform target based encoding of the categorical attributes

# In[1]:


#### Import the necessary modules
## To read and manipulate the data/dataframe
import pandas as pd
import numpy as np

## For Data Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

## For Modelling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, SVR

## Evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


data=pd.read_csv("UnivBank.csv",na_values=['?',"#"])
data.shape


# In[3]:


data.columns


# In[4]:


X=data.drop('Personal Loan',axis=1)
y=data['Personal Loan']


# In[5]:


X_train,X_test,y_train,y_test=train_test_split(data.loc[:,data.columns !='Personal Loan'],data.loc[:,'Personal Loan'],test_size=0.3,random_state=123)


# In[6]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[7]:


X_train.dtypes


# In[8]:


#### Type conversion
to_drop=['ID','ZIP Code']
cat=['Family','Education', 'Securities Account','CD Account', 'Online', 'CreditCard']


# In[9]:


## Type conversion on train
X_train[cat]=X_train[cat].astype('category')


# In[10]:


## Type conversion on test
X_test[cat]=X_test[cat].astype('category')


# In[11]:


X_train.dtypes


# In[12]:


### Dropping attributes
X_train.drop(to_drop,axis=1,inplace=True)
X_test.drop(to_drop,axis=1,inplace=True)


# In[13]:


X_train.dtypes


# In[14]:


X_train.isna().sum()


# In[15]:


###
si_num=SimpleImputer(strategy="mean")
si_cat=SimpleImputer(strategy='most_frequent')


# In[16]:


X_train_num=X_train.drop(cat,axis=1)
X_train_cat=X_train[cat]

### on test
X_test_num=X_test.drop(cat,axis=1)
X_test_cat=X_test[cat]


# In[17]:


X_train_num.dtypes


# In[18]:


X_train_cat.dtypes


# In[19]:


X_train_num=pd.DataFrame(si_num.fit_transform(X_train_num),columns=X_train_num.columns)


# In[20]:


X_train_num.isna().sum()


# In[21]:


X_train_cat=pd.DataFrame(si_cat.fit_transform(X_train_cat),columns=X_train_cat.columns)


# In[22]:


X_train_cat.isna().sum()


# In[23]:


X_test_num=pd.DataFrame(si_num.transform(X_test_num),columns=X_test_num.columns)
X_test_num.isna().sum()


# In[24]:


X_test_cat=pd.DataFrame(si_cat.transform(X_test_cat),columns=X_test_cat.columns)


# In[25]:


### Statndardization of the numeric data
std= StandardScaler()
X_train_num=pd.DataFrame(std.fit_transform(X_train_num),columns= X_train_num.columns)
X_test_num=pd.DataFrame(std.transform(X_test_num),columns=X_test_num.columns)


# In[26]:


X_train_num.head(10)


# In[27]:


## One-hot encoding of categorical data
ohe=OneHotEncoder(handle_unknown='ignore')


# In[28]:


X_train_cat=pd.DataFrame(ohe.fit_transform(X_train_cat).todense(),columns=ohe.get_feature_names_out())


# In[29]:


X_train_cat.head(10)


# In[30]:


X_test_cat=pd.DataFrame(ohe.transform(X_test_cat).todense(),columns=ohe.get_feature_names_out())


# In[31]:


######### Combining Numeric and Categorical Data
Train=pd.concat([X_train_num,X_train_cat],axis=1)


# In[32]:


Train.shape


# In[33]:


Train.head(10)


# In[34]:


Test= pd.concat([X_test_num,X_test_cat],axis=1)


# In[35]:


Test.shape


# In[36]:


### Mod
y_train=y_train.astype('category')
y_test=y_test.astype('category')


# In[37]:


mod=SVC(kernel='rbf',C=2)
mod.fit(Train,y_train)


# In[38]:


preds_train=mod.predict(Train)


# In[39]:


preds_test=mod.predict(Test)


# In[40]:


confusion_matrix(y_train,preds_train)


# In[41]:


confusion_matrix(y_test,preds_test)


# In[42]:


print(classification_report(y_test,preds_test))


# In[ ]:




