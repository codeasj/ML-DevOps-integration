
#importing 
from keras.datasets import mnist

import keras

from keras.models import Sequential

from keras.utils.np_utils import to_categorical

from keras.layers import Dense

from keras.optimizers import Adam

from keras.backend import clear_session

#loading model
(X_train , y_train), (X_test , y_test) = mnist.load_data("mymnist.db")

#reshaping the data
X_test_ = X_test.reshape(-1 , 28*28)
X_train = X_train.reshape(-1 ,  28*28)

# changing the datatype
X_test = X_test.astype("float32")
X_train = X_train.astype("float32")

#performing One hot encoding
y_test= to_categorical(y_test)

y_train= to_categorical(y_train)

#creating model and layers
model = Sequential()

model.add(Dense(units = 40, input_dim = 28*28 , activation = 'relu'))

model.add(Dense(units=150, input_dim = 28*28 , activation = 'relu'))

model.add(Dense(units=60 , input_dim = 28*28 , activation = 'relu'))

model.add(Dense(units=10, input_dim = 28*28 , activation = 'softmax'))


model.compile( optimizer= "Adam" , loss='categorical_crossentropy', 
             metrics=['accuracy'] )

model.summary()

model_history = model.fit(X_train,y_train,epochs=5,verbose=False)

text = model_history.history

accuracy = text['accuracy'][4] * 100


accuracy = int(accuracy)
f= open("accuracy.txt","w+")
f.write(str(accuracy))
f.close()
print("Accuracy for the model is : " , accuracy ,"%")



