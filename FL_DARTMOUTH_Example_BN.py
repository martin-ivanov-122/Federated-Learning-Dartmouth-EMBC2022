#!/usr/bin/env python
# coding: utf-8

# # Install Libraries

# # Import Libraries

# In[2]:


import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

import os 

from fl_utils import *
from functions import *


# # Data Set up

# ## Path to Data & Loading Data

# ### Read the View Text files

# In[3]:



def readviewtextfile (filename):
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    num_par = len (lines)
    num_feat = len (lines[0].split (","))
    feat_matrix = np.zeros ((num_par,num_feat))

    for i in range (num_par):
        clean_items = lines[i].replace ('\n','')
        seperated_items = clean_items.split(",")

        for j in range (num_feat):
            feat_matrix [i][j] = float (seperated_items[j])
    return feat_matrix


# In[4]:


avg_view = readviewtextfile ('avg_view.txt')
loc_view = readviewtextfile ('loc_view.txt')
# trend view
phq_score = readviewtextfile ('phq_score.txt')


# ### Adjustment of the Labels

# In[5]:


label_encoder = preprocessing.LabelEncoder()
 
label_list = label_encoder.fit_transform(phq_severity_sri (phq_score))


# ## TESTING PURPOSES BY COMBINING ALL VIEWS

# In[6]:


ft_matrix = np.hstack ((avg_view,loc_view))


# ## Test-Train Split

# In[136]:


#split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(ft_matrix, 
                                                    label_list, 
                                                    test_size=0.2, 
                                                    random_state=52)

y_train = y_train.reshape (len(y_train),1)
y_test = y_test.reshape (len(y_test),1)


# # Federated Learning Set-up

# ## Create Clients

# In[138]:


#create clients
clients = create_clients(X_train, y_train, num_clients=1, initial='client')


# In[139]:


print (clients)


# ## Batch Cliets

# In[140]:


#process and batch the training data for each client
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data,10)
    


# In[141]:


clients_batched['client_1']


# In[142]:


# for element in clients_batched['client_1']:
#     print (element)


# In[143]:


#process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))


# In[144]:


# for element in test_batched:
#     print (element)


# ## Declearing model optimizer, loss function and a metric

# In[146]:


#Learning Rate
lr = 1 

# number global epochs (aggregations)
comms_round = 10

#Loss Function
loss='categorical_crossentropy'

#Evaluation Metric
metrics = ['accuracy']

#Model optimization using SGD (Stochastic Gradient Descent)
optimizer = SGD(learning_rate=lr, decay=lr / comms_round, momentum=0.9)              


# # Model Aggregation (Federated Averaging) and Testing 

shape = ft_matrix.shape[1] # Num of Feats
classes = 1 # Time series therefore is 1


# In[150]: FL BEGIN


#initialize global model
smlp_global = SimpleMLP()
global_model = smlp_global.build(shape, classes)
        
#commence global training loop
for comm_round in range(comms_round):
            
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    #randomize client data - using keys
    client_names= list(clients_batched.keys())
    random.shuffle(client_names)
    
    #loop through each client and create new local model
    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(shape, classes)
        local_model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=metrics)
        
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        #fit local model with client's data
        local_model.fit(clients_batched[client], epochs=1, verbose=0)
        
        #scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        
        #clear session to free memory after each communication round
        K.clear_session()
        
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    
    #update global model 
    global_model.set_weights(average_weights)

    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        
        global_acc, y_predict = test_model(X_test, Y_test, global_model, comm_round)
        print(Y_test)
        print (y_predict)
        print('comm_round: {} | global_acc: {:.3%}'.format(comm_round, global_acc))



# # In[MLP Testing for Dataset]:
# # X_train, X_test, y_train, y_test
# # SimpleMLP

# MLP_Model = SimpleMLP()
# model = MLP_Model.build(shape, classes)

# # Train the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10, batch_size=5, verbose=1, validation_split=0.2)

# # Evaluate
# test_results = model.evaluate(X_test, Y_test, verbose=1)
# print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')



# # In[SVM Testing for Dataset]:
# import numpy as np
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn import svm

# clf = svm.SVC(kernel ='linear')

# # Fit and Train
# clf.fit(X_train, y_train)

# # Evaluate
# y_pred = clf.predict(X_test)
# acc = accuracy_score(y_pred, Y_test)

# print ('Accuracy: %.2f' %(acc))

# # https://stats.stackexchange.com/questions/39243/how-does-one-interpret-svm-feature-weights









