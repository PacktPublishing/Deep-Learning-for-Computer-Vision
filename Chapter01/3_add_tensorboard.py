import tensorflow as tf

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
z = tf.add(x, y, name='sum')
session = tf.Session()
summary_writer = tf.summary.FileWriter('/Users/i335713/Downloads/logs', session.graph)
summary_writer.flush()

