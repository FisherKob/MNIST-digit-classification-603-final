#!/usr/bin/env python
# coding: utf-8

# #### Importing relevant libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import timeit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation, Flatten, Conv1D,MaxPooling1D
from keras.utils import to_categorical


# #### Loading the datasets

# In[2]:


train=pd.read_csv('/Users/nischal/Downloads/Stat603/Midterm/mnist_train_counts.csv')
test=pd.read_csv('/Users/nischal/Downloads/Stat603/Midterm/mnist_test_counts.csv')
test1=pd.read_csv('/Users/nischal/Downloads/Stat603/Week12/Final  Project 1/mnist_test_counts_new.csv')


# In[3]:


test


# In[4]:


test1


# In[5]:


test


# #### Data Exploration

# In[ ]:





# In[6]:


range(len(train.columns))


# In[7]:


x=[]
for i in range(len(train.columns)):
    if i==0:
        continue
    x.append('x'+str(i))
x


# In[8]:


x.insert(0,'label')
x


# In[9]:


len(x)


# In[10]:


train.columns=x
train


# In[11]:


test.columns=x
test


# In[12]:


test1.columns=x
test1


# In[27]:


plt.figure(figsize=(10,10))
plt.bar(X_test1.columns,X_test1.sum())
plt.xticks(rotation=90)
plt.show()


# In[13]:


test1['label'].unique()


# In[26]:


X_test1=test1.drop('label',axis=1)


# In[15]:


test['label'].unique()


# In[16]:


k5=train[train['label']==5]


# In[17]:


k5


# #### There are 894 rows where label=5

# #### Divide the train/test split

# In[18]:


Y_train=train.pop('label')


# In[19]:


X_train=train


# In[20]:


Y_test=test.pop('label')


# In[21]:


X_test=test


# In[22]:


k5


# #### Define missclassification rate metric

# In[ ]:


def missclassification_rate(x,y):
    z=0
    for i in range(len(Y_test)):
        if x[i]!=y[i]:
            z=z+1
    mis_rate=z/len(Y_test)*100
    return mis_rate
    


# #### Model building

# ##### Multinomial Logistic regression with L2(Ridge) penalty

# In[23]:


logistic_model=LogisticRegression(multi_class='multinomial',penalty='l2')


# In[99]:


get_ipython().run_line_magic('timeit', 'logistic_model.fit(X_train,Y_train)')


# In[25]:


logistic_pred=logistic_model.predict(X_test)


# In[26]:


logistic_pred.shape


# In[27]:


len(Y_test)


# In[29]:


logistic_missclassification_rate=missclassification_rate(Y_test,logistic_pred)
logistic_missclassification_rate


# In[30]:


logistic_accuracy=100-logistic_missclassification_rate
logistic_accuracy


# In[31]:


logistic_pred1=logistic_model.predict(X_test1)
logistic_pred1


# #### Random Forest with GridSearchCV technique for hyperparameter tuning 

# In[32]:


hyperparameters={'n_estimators':[80,100,120],'max_depth':[None,4]}


# In[33]:


rdf=RandomForestClassifier()


# In[34]:


random_forest_model=GridSearchCV(rdf,hyperparameters)


# In[103]:


get_ipython().run_line_magic('timeit', 'random_forest_model.fit(X_train,Y_train)')


# In[36]:


rf_predict=random_forest_model.predict(X_test)


# In[37]:


rf_missclassification_rate=missclassification_rate(Y_test,rf_predict)
rf_missclassification_rate


# In[38]:


rf_accuracy=100-rf_missclassification_rate
rf_accuracy


# In[39]:


rf_predict1=random_forest_model.predict(X_test1)


# In[40]:


rf_missclassification_rate1=missclassification_rate(Y_test,rf_predict1)
rf_missclassification_rate1


# #### Multiclass Support Vector Machine with GridSearchCV

# In[41]:


hyperparameters={'kernel':['linear','poly','sigmoid'],'degree':[2,3],'gamma':['auto']}


# In[42]:


svm=SVC()


# In[43]:


support_vector_model=GridSearchCV(svm,hyperparameters)


# In[104]:


get_ipython().run_line_magic('timeit', 'support_vector_model.fit(X_train,Y_train)')


# In[45]:


support_vector_model.best_estimator_


# In[46]:


svm_predict=support_vector_model.predict(X_test)
svm_predict


# In[47]:


svm_misclassification_rate=missclassification_rate(Y_test,svm_predict)
svm_misclassification_rate


# In[48]:


svm_accuracy=100-svm_misclassification_rate
svm_accuracy


# In[49]:


svm_predict1=support_vector_model.predict(X_test1)
svm_predict1


# #### Gradient Boosing with GridSearchCV

# In[50]:


hyperparameters={'loss':['deviance'],'learning_rate':[0.05,0.1],'n_estimators':[120],'criterion':['mse']}


# In[51]:


GB=GradientBoostingClassifier()


# In[52]:


gradient_boosing_model=GridSearchCV(GB,hyperparameters)


# In[102]:


get_ipython().run_line_magic('timeit', 'gradient_boosing_model.fit(X_train,Y_train)')


# In[53]:


gb_predict=gradient_boosing_model.predict(X_test)
gb_predict


# In[54]:


gb_misclassification_rate=missclassification_rate(Y_test,svm_predict)
gb_misclassification_rate


# In[55]:


gb_accuracy=100-rf_missclassification_rate
gb_accuracy


# In[56]:


gb_predict1=gradient_boosing_model.predict(X_test1)
gb_predict1


# #### Multi-class Linear Discriminant Analysis (LDA) with GridSearchCV

# In[57]:


hyperparameters={'solver':['svd','lsqr'],'shrinkage':[None]}


# In[58]:


LDA=LinearDiscriminantAnalysis()


# In[59]:


linear_discriminant_model=GridSearchCV(LDA,hyperparameters)


# In[101]:


get_ipython().run_line_magic('timeit', 'linear_discriminant_model.fit(X_train,Y_train)')


# In[60]:


lda_predict=linear_discriminant_model.predict(X_test)
lda_predict


# In[61]:


lda_misclassification_rate=missclassification_rate(Y_test,lda_predict)
lda_misclassification_rate


# In[63]:


lda_accuracy=100-rf_missclassification_rate
lda_accuracy


# In[64]:


lda_predict1=linear_discriminant_model.predict(X_test1)
lda_predict1


# #### Convolutional Neural Network 

# In[65]:


X_train.shape


# In[66]:


Y_train.shape


# In[67]:


num_classes=10
y_train=to_categorical(Y_train,num_classes)
y_test=to_categorical(Y_test,num_classes)


# In[68]:


X_train.shape


# In[69]:


model=Sequential()
model.add(Conv1D(32,kernel_size=(3),input_shape=(49,1,),activation='relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Conv1D(32,kernel_size=(3),activation='relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Dropout(0.1))
model.add(Flatten())


model.add(Dense(units=(64),activation='relu'))

model.add(Dense(num_classes,activation='softmax'))


# In[70]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[71]:


model.summary()


# In[72]:


X_train = X_train.to_numpy()


# In[73]:


X_test1=X_test1.to_numpy()


# In[74]:


NN_train = X_train.reshape(-1,49,1)


# In[75]:


y_train.shape


# In[76]:


X_test = X_test.to_numpy()


# In[77]:


NN_test = X_test.reshape(-1,49,1)


# In[78]:


NN_test1=X_test1.reshape(-1,49,1)


# In[79]:


y_train.shape


# In[100]:


batch_size=32
epochs=10

get_ipython().run_line_magic('timeit', 'model.fit(NN_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(NN_test,y_test))')
score=model.evaluate(NN_test,y_test)


# In[81]:


CNN_pred=model.predict(NN_test)


# In[82]:


CNN_pred1=tf.argmax(CNN_pred, axis=-1)
CNN_pred1=CNN_pred1.numpy()
CNN_pred1


# In[83]:


CNN_pred2=model.predict(NN_test1)
CNN_pred2


# In[84]:


CNN_pred2=tf.argmax(CNN_pred2, axis=-1)
CNN_pred2=CNN_pred2.numpy()
CNN_pred2


# #### Summary

# In[85]:


list(CNN_pred1)


# In[86]:


pd.Series(list(CNN_pred1))


# In[87]:


CNN_pred


# In[88]:


CNN_pred1


# In[89]:


cnn_missclassification_rate=missclassification_rate(Y_test,CNN_pred1)
cnn_missclassification_rate


# In[90]:


accuracy=100-cnn_missclassification_rate
accuracy


# In[91]:


label_predict=pd.DataFrame({'CNN':list(CNN_pred1),'LDA':list(lda_predict),'GB':list(gb_predict),'SVM':list(svm_predict),'RF':list(rf_predict),'Logistic':list(logistic_pred)})
label_predict



# In[92]:


label_predict1=pd.DataFrame({'CNN':list(CNN_pred2),'LDA':list(lda_predict1),'GB':list(gb_predict1),'SVM':list(svm_predict1),'RF':list(rf_predict1),'Logistic':list(logistic_pred1)})
label_predict1



# In[93]:


#label_predict1.to_csv('myfinal_prediction2.csv')


# In[94]:


#label_predict.to_csv('myfinal_prediction1.csv')


# In[95]:


missclasification_rate={'GB':gb_misclassification_rate,'LDA':lda_misclassification_rate,
                        'SVM':svm_misclassification_rate,'RF':rf_missclassification_rate,
                        'Logistic':logistic_missclassification_rate,'CNN':cnn_missclassification_rate}
                        


# In[96]:


missclasification_rate


# In[ ]:





# In[97]:


plt.figure(figsize=(10,7))
plt.bar(range(len(missclasification_rate)), list(missclasification_rate.values()), align='center')
plt.xticks(range(len(missclasification_rate)), list(missclasification_rate.keys()))
plt.xticks(rotation=45)
plt.title('Missclassification Rate of models')
plt.ylabel('Missclassification Rate %')
plt.xlabel('Models')
plt.show()


# In[105]:


timeit={'GB': 240000,'LDA':663 ,'SVM' :150000 ,'RF':26200 ,'Logistic':461, 'CNN':6870}


# In[109]:


plt.figure(figsize=(10,7))
plt.bar(range(len(timeit)), list(timeit.values()), align='center')
plt.xticks(range(len(missclasification_rate)), list(missclasification_rate.keys()))
plt.xticks(rotation=45)
plt.title('Model train time in Milliseconds(ms)')
plt.ylabel('timeit')
plt.xlabel('Milliseconds(ms)')
plt.show()


# In[ ]:




