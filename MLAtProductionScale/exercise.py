#This code is that I use Estimator and implement text classification with movie reviews written in Learn and use ML section

import tensorflow as tf
import numpy as np

imdb = tf.keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = imdb.load_data()
"""
Train_l=[]
Test_l=[]
for x in train_labels:
    Train_l.append([x])
for x in test_labels:
    Test_l.append([x])
train_labels=np.array(Train_l)
test_labels=np.array(Test_l)
"""
train_data=tf.keras.preprocessing.sequence.pad_sequences(train_data,maxlen=260,padding="post",truncating="post",dtype="float32")
test_data=tf.keras.preprocessing.sequence.pad_sequences(test_data,maxlen=260,padding="post",truncating="post",dtype="float32")

#jprint(len(train_data))
#print(len(train_labels))
#print(len(test_labels))
#print(len(test_data))
def model_fn(features,labels,mode):
    #Model level
    input_layer = features["x"]
    hidden_layer1 = tf.layers.dense(inputs=input_layer,activation=tf.nn.relu,units=100)
    hidden_layer2 = tf.layers.dense(inputs=hidden_layer1,activation=tf.nn.relu,units=100)
    logits=tf.layers.dense(inputs=hidden_layer2,units=2)
    print(logits)
    predictions = {
            "classes" : tf.argmax(logits,axis=1)
            }
    if mode== tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    train_op=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss,tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,train_op=train_op,loss=loss)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops={
                "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

train_input_fn=tf.estimator.inputs.numpy_input_fn(
    x={"x":train_data},
    y=train_labels,
    num_epochs=100,
    shuffle=False,
    batch_size=1
    )
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":test_data},
    y=test_labels,
    shuffle=False,
    batch_size=1
    )
classifier = tf.estimator.Estimator(model_fn=model_fn)
classifier.train(input_fn=train_input_fn,steps=20)
res = classifier.evaluate(input_fn=eval_input_fn)
print(res)
