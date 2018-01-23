import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
session = tf.Session()
print(session.run(hello))