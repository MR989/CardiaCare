#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score


# In[37]:


#import plotly

#resding dataset


# In[38]:


df=pd.read_csv(r'W:\code\code\FYP_HeartDisease\MLModel\dataset.csv')
df


# In[39]:


#numerical analysis

df['target']
df.groupby('target').size()


# In[40]:


df.shape


# In[41]:


df.size


# In[42]:


df.describe()


# In[43]:


#check if any null values

df.info()


# In[ ]:




    


# In[44]:


#visualization


# In[ ]:





# In[ ]:





# In[46]:


df.hist(figsize=(14,14))
plt.show()


# In[ ]:





# # data preprocessing 

# divide the dataset in to x and y such that x contains all the features and y contains the output

# In[ ]:





# In[47]:


x,y=df.loc[:,:'thal'],df['target']

#loc method is used to call the columns by its name
#loc used to Access a group of rows and columns by label(s) or a boolean array.
#ilock method is used to call the columns by its index value

x


# In[48]:


y


# In[49]:


# dividing the dataset for training and testing (Split arrays or matrices into random train and test subsets)


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
x


# In[52]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3,test_size=0.3,shuffle=True)


# In[53]:


x_train


# x_test

# In[ ]:





# # making the model with suitable algorithms
# 

# #Decission Tree classifier

# In[54]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=7)
dt.fit(x_train,y_train)


# In[55]:


x_test


# In[56]:


prediction=dt.predict(x_test)


# In[57]:


prediction


# In[58]:


y_test


# In[59]:


accuracy_dt=accuracy_score(y_test,prediction)*100


# In[60]:


accuracy_dt


# In[61]:


dt.feature_importances_


#  Plotting the most imp feature
#  

# In[62]:


# def plot_most_important_feature(model):
#     plt.figure(figsize=(8,6))
#     n_features=13
#     plt.barh(range(n_features),model.feature_importances_,align='center')
    
#     plt.yticks(np.arange(n_features),x)
#     plt.xlabel("Feature Importance")
#     plt.ylabel("Features")
#     plt.ylim(-1,n_features)
    
    
# plot_most_important_feature(dt)


# In[ ]:




# Predicting using Custom Data
    

# In[98]:


category=["Your heart is healthy",'Yes, you have a heart disease ']


# In[113]:


custom_data=np.array([[52,1,1,120,325,0,1,172,0,0.2,2,0,2]])
custom_data_prediction_dt=dt.predict(custom_data)
custom_data_prediction_dt
print(category[int(custom_data_prediction_dt)])


# In[ ]:





# In[ ]:





# #K Nearest Neighbour
# 

# In[30]:


from sklearn.neighbors import KNeighborsClassifier


# In[31]:


#Finding the best value of k
k_range=range(1,26)
scores={}
h_score = 0       # to find the best score
best_k=0          # to find the best k
scores_list=[]  

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    prediction_knn=knn.predict(x_test)
    scores[k]=accuracy_score(y_test,prediction_knn)
    if scores[k]>h_score:
        h_score = scores[k]
        best_k = k

    scores_list.append(accuracy_score(y_test,prediction_knn))
print('The best value of k is {} with score : {}'.format(best_k,h_score))


# In[ ]:





# In[105]:




knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)

prediction_knn=knn.predict(x_test)


# In[106]:


accuracy_knn=accuracy_score(y_test,prediction_knn)*100
accuracy_knn


# In[ ]:





# In[112]:


custom_data=np.array([[52,1,1,120,325,0,1,172,0,0.2,2,0,2]])
custom_data_prediction_knn=knn.predict(custom_data)
print(custom_data_prediction_knn)
print(category[int(custom_data_prediction_knn)])


# In[ ]:





# In[ ]:





# In[ ]:





# In[66]:


# Which algo did best


# In[35]:


algorithms=["Decission Tree","KNN"]
scores=[accuracy_dt,accuracy_knn]

plt.bar(algorithms,scores)




if(accuracy_dt>accuracy_knn):
    moddel=dt
else:
    moddel=knn


filename="final_model.sav"
joblib.dump(moddel,filename)





# In[ ]: