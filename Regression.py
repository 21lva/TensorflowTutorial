#Aim : predict the median prices of homes in a Boston suburb druing the mid-1970s
import tensorflow as tf
from tensorflow import keras
import numpy as np

boston_housing=keras.datasets.boston_housing

(train_data,train_labels),(test_data,test_labesl) = boston_housing.load_data()

order = np.argsort(np.random.random(train_labels.shape))
train_data=train_data[order]
train_labels=train_labels[order]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data-mean)/std
test_data = (test_data-mean)/std

class M():
    def __init__(self,numLayer):
        self.model=keras.Sequential()
        self.model.add(keras.layers.Dense(64,activation=tf.nn.relu,input_shape=(train_data.shape[1],)))
        for _ in range(numLayer):
            self.model.add(keras.layers.Dense(64,activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(1))
        self.optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001)
        self.model.compile(loss="mse",optimizer=self.optimizer,weighted_metrics=["mae"])

    def train(self,X,Y):
        self.model.fit(X,Y,batch_size=500,validation_split=0.2,callbacks=[keras.callbacks.EarlyStopping(patience=20)])


    def predict(self,X):
        return self.model.predict(X)

    def evaluate(self,X,Y):
        return self.model.evaluate(X,Y,verbose=0)
myModel=M(1)
myModel.train(train_data,train_labels)

loss,mae=myModel.evaluate(test_data,test_labesl)
print("loss: ",loss," mae : ",mae)
