import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
#The num_words means that
# I will use 10000 most frequently occuring words in the training data.

#print(train_data[0])
word_index=imdb.get_word_index()
word_index={k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"]=0
word_index["<START>"]=1
word_index["<UNK>"]=2
word_index["<UNUSED>"]=3
reverse_word_index = {value:key for (key,value) in word_index.items()}

def num_to_words(text):
    return " ".join([reverse_word_index.get(i,"??")for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding="post",maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding="post",maxlen=256)

vocab_size=10000

model=keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.compile(optimizer=tf.train.AdamOptimizer(),loss="binary_crossentropy",weighted_metrics=["accuracy"])
#you have to use weighted_metrices instead of metrices written in the tutorial text.
model.fit(train_data,train_labels,epochs=40,batch_size=512,verbose=1)

result = model.evaluate(test_data,test_labels)
print(result)





