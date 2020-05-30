#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing 

from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adam
from keras.backend import clear_session
import numpy
import keras


# In[ ]:


#defining the model function

def defmodel(neurons , model , epochs , test) : 
	model.add(Dense(units = neurons , input_dim = 28*28 , activation = 'relu'))
	model.add(Dense(units=150 , input_dim = 28*28 , activation = 'relu'))
	model.add(Dense(units=80 , input_dim = 28*28 , activation = 'relu'))
	model.add(Dense(units=10 , input_dim = 28*28 , activation = 'softmax'))
	model.compile( optimizer= "Adam" , loss='categorical_crossentropy', 
	             metrics=['accuracy'] )
	return model



# In[ ]:


# fitting the model
def fitmodel(fit_model, epochs):
	text = fit_model.history
	accuracy = text['accuracy'][epochs-1] * 100
	accuracy = int(accuracy)
	f= open("accuracy.txt","w+")
	f.write(str(accuracy))
	f.close()
	print("    Accuracy for this change in model  : " , accuracy ,"%")
	return accuracy




# In[ ]:


# Load Model 
(X_train , y_train), (X_test ,y_test) = mnist.load_data("mymnist.db")


# In[ ]:


# Reshape data 
X_test = X_test.reshape(-1 , 28*28)
X_train = X_train.reshape(-1 ,  28*28)
#changing data type
X_test = X_test.astype("float32")
X_train = X_train.astype("float32")


# In[ ]:


# One hot encoding 
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)



# In[ ]:


#predefining
neurons = 10
accuracy = 0
epochs = 1
test = 1
flag = 0


# In[ ]:


#running while the defined accuracy is achieved
while int(accuracy) < 98:
	if flag == 1 :
		model = keras.backend.clear_session()
		neurons = neurons+10
		epochs = epochs+1 
		test = test + 1
	model = Sequential()
	model = defmodel(neurons , model , epochs , test)
	print(" Finding accuracy . . .")
	fit_model = model.fit(X_train ,  y_train , epochs = epochs , verbose =  False)
	accuracy=fitmodel(fit_model , epochs)
	flag = 1


# In[ ]:




