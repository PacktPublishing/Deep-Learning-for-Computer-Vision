import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

input_size = 784
no_classes = 10
batch_size = 100
total_batches = 100

x_input = tf.placeholder(tf.float32, shape=[None, input_size])
y_input = tf.placeholder(tf.float32, shape=[None, no_classes])


def add_variable_summary(tf_variable, summary_name):
  with tf.name_scope(summary_name + '_summary'):
    mean = tf.reduce_mean(tf_variable)
    tf.summary.scalar('Mean', mean)
    with tf.name_scope('standard_deviation'):
        standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))
    tf.summary.scalar('StandardDeviation', standard_deviation)
    tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
    tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
    tf.summary.histogram('Histogram', tf_variable)


x_input_reshape = tf.reshape(x_input, [-1, 28, 28, 1], name='input_reshape')

convolution_layer_1 = tf.layers.conv2d(
  inputs=x_input_reshape,
  filters=64,
  kernel_size=[3, 3],
  activation=tf.nn.relu,
)
add_variable_summary(convolution_layer_1, 'convolution1')
pooling_layer_1 = tf.layers.max_pooling2d(
    inputs=convolution_layer_1,
    pool_size=[2, 2],
    strides=2
)
add_variable_summary(pooling_layer_1, 'pooling1')


convolution_layer_2 = tf.layers.conv2d(
  inputs=pooling_layer_1,
  filters=128,
  kernel_size=[3, 3],
  activation=tf.nn.relu,
)
add_variable_summary(convolution_layer_2, 'convolution2')
pooling_layer_2 = tf.layers.max_pooling2d(
    inputs=convolution_layer_2,
    pool_size=[2, 2],
    strides=2
)
add_variable_summary(pooling_layer_2, 'pool2')

flattened_pool = tf.reshape(pooling_layer_2, [-1, 5 * 5 * 128], name='flattened_pool')

dense_layer = tf.layers.dense(
    inputs=flattened_pool,
    units=1024,
    activation=tf.nn.relu,
    name='dense'
)
add_variable_summary(dense_layer, 'dense')

dropout_bool = tf.placeholder(tf.bool)
dropout_layer = tf.layers.dropout(
    inputs=dense_layer,
    rate=0.4,
    training=dropout_bool,
    name='dropout'
)

logits = tf.layers.dense(inputs=dropout_layer, units=no_classes, name='logits')
add_variable_summary(logits, 'logits')

with tf.name_scope('loss'):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,
                                                                    logits=logits)
    loss_operation = tf.reduce_mean(softmax_cross_entropy, name='loss')
    tf.summary.scalar('loss', loss_operation)

with tf.name_scope('optimiser'):
    optimiser = tf.train.AdamOptimizer().minimize(loss_operation)


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        predictions = tf.argmax(logits, 1)
        correct_predictions = tf.equal(predictions, tf.argmax(y_input, 1))
    with tf.name_scope('accuracy'):
        accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
tf.summary.scalar('accuracy', accuracy_operation)

session = tf.Session()

# Adding the variable in between creating the session and initialising the graph
no_embedding_data = 1000
embedding_variable = tf.Variable(tf.stack(
    mnist.test.images[:no_embedding_data], axis=0), trainable=False)
session.run(tf.global_variables_initializer())

merged_summary_operation = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter('/tmp/train', session.graph)

test_images, test_labels = mnist.test.images, mnist.test.labels

for batch_no in range(total_batches):
    image_batch = mnist.train.next_batch(100)
    _, merged_summary = session.run([optimiser, merged_summary_operation], feed_dict={
        x_input: image_batch[0],
        y_input: image_batch[1],
        dropout_bool: True
    })
    train_summary_writer.add_summary(merged_summary, batch_no)

work_dir = ''  # change path
metadata_path = '/tmp/train/metadata.tsv'

with open(metadata_path, 'w') as metadata_file:
    for i in range(no_embedding_data):
        metadata_file.write('{}\n'.format(
            np.nonzero(mnist.test.labels[::1])[1:][0][i]))

from tensorflow.contrib.tensorboard.plugins import projector
projector_config = projector.ProjectorConfig()
embedding_projection = projector_config.embeddings.add()
embedding_projection.tensor_name = embedding_variable.name
embedding_projection.metadata_path = metadata_path
embedding_projection.sprite.image_path = os.path.join(work_dir + '/mnist_10k_sprite.png')
embedding_projection.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(train_summary_writer, projector_config)
tf.train.Saver().save(session, '/tmp/train/model.ckpt', global_step=1)
