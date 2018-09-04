import numpy as np
import tensorflow as tf
from tensorflow import keras

(train_data,trainl_labels),(test_data,test_labels) =keras.datasets.imdb.load_data()

def multi_hot_seqeuences(sequences,dimension):
    results=np.zeros((len(sequences),dimension))
    for i,word_indices in enumerate(train_data):
        results[i,word_indices]=1.0
    return results

train_data=multi_hot_seqeuences(train_data)
test_data=multi_hot_seqeuences(test_data)
NUMWORDS=10000
baseline_model = keras.Sequential(
    keras.layers.Dense(16,activation=tf.nn.relu,input_shape=(NUMWORDS,),kernel_regularizer=keras.regularizers.l2(0.02)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
)
baseline_model.compile(optimizer="adam",loss="binary_crossentropy",metrices=["accuracy","binary_crossentropy"])
baseline_model.fit(train_data,trainl_labels,epochs=20,batch_size=512,validation_data=(test_data,test_labels),verbose=2)



