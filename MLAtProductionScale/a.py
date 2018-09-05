import tensorflow as tf
import numpy as np
x=tf.constant([[-1],[2.3]])
y=tf.cast(x>0.5,tf.int32)
with tf.Session() as sess:
    k = sess.run(y)
    print(k)
    print(k[0][0])
    print(type(k[0][0]))
