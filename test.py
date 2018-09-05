import tensorflow as tf
import numpy as np

def model_fn(features,labels,mode):
    print("x: ",features["x"])
    print("y: ",features["y"])

cl=tf.estimator.Estimator(model_fn=model_fn)
print("hello")
train_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={"x":np.array([1,2,3]),"y":np.array(["a","b","c"])},
        shuffle=False
        )
cl.train(input_fn=train_input_fn)
