import edward as ed
import tensorflow as tf
from keras.layers import Dropout

x = tf.placeholder(tf.float32, [1, 5])
y = Dropout(0.1)(x)

grad = tf.gradients(y, [x])  # succeeds

y_copy = ed.copy(y)
grads_copy = tf.gradients(y_copy, [x])  # fails
