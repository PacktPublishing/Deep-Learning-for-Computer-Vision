
import tensorflow as tf
import numpy as np
import os
from iou import calculate_iou
from pascal_voc import pascal_voc


DATA_PATH = 'data'
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')
WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weight')
FLIPPED = True
DISP_CONSOLE = False

GPU = ''


THRESHOLD = 0.2

IOU_THRESHOLD = 0.5

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
num_class = len(classes)
image_size = 448
cell_size = 7
boxes_per_cell = 2
output_size = (cell_size * cell_size) * (num_class + boxes_per_cell * 5)
scale = 1.0 * image_size / cell_size
boundary1 = cell_size * cell_size * num_class
boundary2 = boundary1 + cell_size * cell_size * boxes_per_cell

object_scale = 1.0
noobject_scale = 1.0
class_scale = 2.0
coord_scale = 5.0

learning_rate = 0.0001
batch_size = 45
alpha = 0.1

weights_file = None
max_iter = 15000
initial_learning_rate = 0.0001
decay_steps = 30000
decay_rate = 0.1
staircase = True



offset = np.transpose(np.reshape(np.array(
    [np.arange(cell_size)] * cell_size * boxes_per_cell),
    (boxes_per_cell, cell_size, cell_size)), (1, 2, 0))

images = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name='images')


def add_variable_summary(tf_variable, summary_name):
  with tf.name_scope(summary_name + '_summary'):
    mean = tf.reduce_mean(tf_variable)
    tf.summary.scalar('Mean', mean)
    with tf.name_scope('standard_deviation'):
        standard_deviation = tf.sqrt(tf.reduce_mean(
            tf.square(tf_variable - mean)))
    tf.summary.scalar('StandardDeviation', standard_deviation)
    tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
    tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
    tf.summary.histogram('Histogram', tf_variable)

def pooling_layer(input_layer, pool_size=[2, 2], strides=2, padding='valid'):
    layer = tf.layers.max_pooling2d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides,
        padding=padding
    )
    add_variable_summary(layer, 'pooling')
    return layer

def convolution_layer(input_layer, filters, kernel_size=[3, 3], padding='valid',
                      activation=tf.nn.leaky_relu):
    layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        weights_regularizer=tf.l2_regularizer(0.0005)
    )
    add_variable_summary(layer, 'convolution')
    return layer


def dense_layer(input_layer, units, activation=tf.nn.leaky_relu):
    layer = tf.layers.dense(
        inputs=input_layer,
        units=units,
        activation=activation,
        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        weights_regularizer=tf.l2_regularizer(0.0005)
    )
    add_variable_summary(layer, 'dense')
    return layer


yolo = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
yolo = convolution_layer(yolo, 64, 7, 2)
yolo = pooling_layer(yolo, [2, 2], 2, 'same')
yolo = convolution_layer(yolo, 192, 3)
yolo = pooling_layer(yolo, 2, 'same')
yolo = convolution_layer(yolo, 128, 1)
yolo = convolution_layer(yolo, 256, 3)
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = pooling_layer(yolo, 2, 'same')
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = convolution_layer(yolo, 512, 1)
yolo = convolution_layer(yolo, 1024, 3)
yolo = pooling_layer(yolo, 2)
yolo = convolution_layer(yolo, 512, 1)
yolo = convolution_layer(yolo, 1024, 3)
yolo = convolution_layer(yolo, 512, 1)
yolo = convolution_layer(yolo, 1024, 3)
yolo = convolution_layer(yolo, 1024, 3)
yolo = tf.pad(yolo, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]))
yolo = convolution_layer(yolo, 1024, 3, 2)
yolo = convolution_layer(yolo, 1024, 3)
yolo = convolution_layer(yolo, 1024, 3)
yolo = tf.transpose(yolo, [0, 3, 1, 2])
yolo = tf.layers.flatten(yolo)
yolo = dense_layer(yolo, 512)
yolo = dense_layer(yolo, 4096)

dropout_bool = tf.placeholder(tf.bool)
yolo = tf.layers.dropout(
        inputs=yolo,
        rate=0.4,
        training=dropout_bool
    )
yolo = dense_layer(yolo, output_size, None)

predicts = yolo
labels = tf.placeholder(tf.float32, [None, cell_size, cell_size, 5 + num_class])

predict_classes = tf.reshape(predicts[:, :boundary1], [batch_size, cell_size, cell_size, num_class])
predict_scales = tf.reshape(predicts[:, boundary1:boundary2], [batch_size, cell_size, cell_size, boxes_per_cell])
predict_boxes = tf.reshape(predicts[:, boundary2:], [batch_size, cell_size, cell_size, boxes_per_cell, 4])

response = tf.reshape(labels[:, :, :, 0], [batch_size, cell_size, cell_size, 1])
boxes = tf.reshape(labels[:, :, :, 1:5], [batch_size, cell_size, cell_size, 1, 4])
boxes = tf.tile(boxes, [1, 1, 1, boxes_per_cell, 1]) / image_size
classes = labels[:, :, :, 5:]

offset = tf.constant(offset, dtype=tf.float32)
offset = tf.reshape(offset, [1, cell_size, cell_size, boxes_per_cell])
offset = tf.tile(offset, [batch_size, 1, 1, 1])
predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / cell_size,
                               (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / cell_size,
                               tf.square(predict_boxes[:, :, :, :, 2]),
                               tf.square(predict_boxes[:, :, :, :, 3])])
predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

iou_predict_truth = calculate_iou(predict_boxes_tran, boxes)

object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

boxes_tran = tf.stack([boxes[:, :, :, :, 0] * cell_size - offset,
                       boxes[:, :, :, :, 1] * cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                       tf.sqrt(boxes[:, :, :, :, 2]),
                       tf.sqrt(boxes[:, :, :, :, 3])])
boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

class_delta = response * (predict_classes - classes)
class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * class_scale

object_delta = object_mask * (predict_scales - iou_predict_truth)
object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * object_scale

noobject_delta = noobject_mask * predict_scales
noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * noobject_scale

coord_mask = tf.expand_dims(object_mask, 4)
boxes_delta = coord_mask * (predict_boxes - boxes_tran)
coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * coord_scale

tf.losses.add_loss(class_loss)
tf.losses.add_loss(object_loss)
tf.losses.add_loss(noobject_loss)
tf.losses.add_loss(coord_loss)

total_loss = tf.losses.get_total_loss()

yolo = yolo
data = data

global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
learning_rate = tf.train.exponential_decay(
    initial_learning_rate, global_step, decay_steps,
    decay_rate, staircase, name='learning_rate')
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(
    yolo.total_loss, global_step=global_step)
ema = tf.train.ExponentialMovingAverage(decay=0.9999)
averages_op = ema.apply(tf.trainable_variables())
with tf.control_dependencies([optimizer]):
    train_op = tf.group(averages_op)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(1, max_iter + 1):
    images, labels = data.get()
    feed_dict = {yolo.images: images, yolo.labels: labels}
    sess.run(train_op, feed_dict=feed_dict)




