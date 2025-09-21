#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[5]:


#loading the dataset  to a pandas dataframe
diabetes_dataset = pd.read_csv('diabetes.csv')


# In[6]:



diabetes_dataset.info()


# In[7]:


diabetes_dataset.head()


# In[8]:


diabetes_dataset.shape


# In[9]:


diabetes_dataset.describe()


# In[10]:


diabetes_dataset['Outcome'].value_counts()


# In[11]:


diabetes_dataset.groupby('Outcome').mean()


# In[12]:


X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']


# In[13]:


print(X)


# In[14]:


print(Y)


# In[17]:


scalar=StandardScaler()


# In[18]:


scalar.fit(X)


# In[19]:


standaelisted_data=scalar.transform(X)


# In[20]:


X=standaelisted_data
Y=diabetes_dataset['Outcome']


# In[21]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[22]:


print(X.shape,X_train.shape,X_test.shape)


# In[23]:


classifier=svm.SVC(kernel='linear')


# In[24]:


classifier.fit(X_train,Y_train)


# In[25]:


X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[30]:


print("checking the accuracy of the trained data",training_data_accuracy)


# In[27]:


X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[28]:


print("Prediction of the test data".test_data_accuracy)


# In[29]:


# makeing the prediction data
input_data=( 1,85,66,29,0,26.6,0.351,31)

input_data_as_numpy_array=np.asarray(input_data)

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

std_data=scalar.transform(input_data_reshaped)
print(std_data)

predicition=classifier.predict(std_data)
print(predicition)

if(predicition[0]==0):
  print('the person is not diabetic')
else:
  print('the person is diabetic')


# In[ ]:




