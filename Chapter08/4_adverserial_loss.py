import tensorflow as tf

batch_size = 32
input_dimension = [227, 227]
real_images = None
labels = None


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


def convolution_layer(input_layer,
                      filters,
                      kernel_size=[4, 4],
                      activation=tf.nn.leaky_relu):
    layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        kernel_regularizer=tf.nn.l2_loss,
        bias_regularizer=tf.nn.l2_loss,
    )
    add_variable_summary(layer, 'convolution')
    return layer


def transpose_convolution_layer(input_layer,
                                filters,
                                kernel_size=[4, 4],
                                activation=tf.nn.relu,
                                strides=2):
    layer = tf.layers.conv2d_transpose(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        strides=strides,
        kernel_regularizer=tf.nn.l2_loss,
        bias_regularizer=tf.nn.l2_loss,
    )
    add_variable_summary(layer, 'convolution')
    return layer

def pooling_layer(input_layer,
                  pool_size=[2, 2],
                  strides=2):
    layer = tf.layers.max_pooling2d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides
    )
    add_variable_summary(layer, 'pooling')
    return layer


def dense_layer(input_layer,
                units,
                activation=tf.nn.relu):
    layer = tf.layers.dense(
        inputs=input_layer,
        units=units,
        activation=activation
    )
    add_variable_summary(layer, 'dense')
    return layer


def get_generator(input_noise, is_training=True):
    generator = dense_layer(input_noise, 1024)
    generator = tf.layers.batch_normalization(generator, training=is_training)
    generator = dense_layer(generator, 7 * 7 * 256)
    generator = tf.layers.batch_normalization(generator, training=is_training)
    generator = tf.reshape(generator,  [-1, 7, 7, 256])
    generator = transpose_convolution_layer(generator, 64)
    generator = tf.layers.batch_normalization(generator, training=is_training)
    generator = transpose_convolution_layer(generator, 32)
    generator = tf.layers.batch_normalization(generator, training=is_training)
    generator = convolution_layer(generator, 3)
    generator = convolution_layer(generator, 1, activation=tf.nn.tanh)
    print(generator)
    return generator


def get_discriminator(image, is_training=True):
    x_input_reshape = tf.reshape(image, [-1, 28, 28, 1],
                                 name='input_reshape')
    discriminator = convolution_layer(x_input_reshape, 64)
    discriminator = convolution_layer(discriminator, 128)
    discriminator = tf.layers.flatten(discriminator)
    discriminator = dense_layer(discriminator, 1024)
    discriminator = tf.layers.batch_normalization(discriminator, training=is_training)
    discriminator = dense_layer(discriminator, 2)
    return discriminator

def fully_connected_layer(input_layer, units):
    return tf.layers.dense(
        input_layer,
        units=units,
        activation=tf.nn.relu
    )


def convolution_layer(input_layer, filter_size):
    return  tf.layers.conv2d(
        input_layer,
        filters=filter_size,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        kernel_size=3,
        strides=2
    )


def deconvolution_layer(input_layer, filter_size, activation=tf.nn.relu):
    return tf.layers.conv2d_transpose(
        input_layer,
        filters=filter_size,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        kernel_size=3,
        activation=activation,
        strides=2
    )

def get_autoencoder():
    input_layer = tf.placeholder(tf.float32, [None, 128, 128, 3])
    convolution_layer_1 = convolution_layer(input_layer, 1024)
    convolution_layer_2 = convolution_layer(convolution_layer_1, 512)
    convolution_layer_3 = convolution_layer(convolution_layer_2, 256)
    convolution_layer_4 = convolution_layer(convolution_layer_3, 128)
    convolution_layer_5 = convolution_layer(convolution_layer_4, 32)

    convolution_layer_5_flattened = tf.layers.flatten(convolution_layer_5)
    bottleneck_layer = fully_connected_layer(convolution_layer_5_flattened, 16)
    c5_shape = convolution_layer_5.get_shape().as_list()
    c5f_flat_shape = convolution_layer_5_flattened.get_shape().as_list()[1]
    fully_connected = fully_connected_layer(bottleneck_layer, c5f_flat_shape)
    fully_connected = tf.reshape(fully_connected,
                                 [-1, c5_shape[1], c5_shape[2], c5_shape[3]])

    deconvolution_layer_1 = deconvolution_layer(fully_connected, 128)
    deconvolution_layer_2 = deconvolution_layer(deconvolution_layer_1, 256)
    deconvolution_layer_3 = deconvolution_layer(deconvolution_layer_2, 512)
    deconvolution_layer_4 = deconvolution_layer(deconvolution_layer_3, 1024)
    deconvolution_layer_5 = deconvolution_layer(deconvolution_layer_4, 3,
                                                activation=tf.nn.tanh)
    return deconvolution_layer_5

gan = tf.contrib.gan.gan_model(
    get_autoencoder,
    get_discriminator,
    real_images,
    real_images)

loss = tf.contrib.gan.gan_loss(
    gan, gradient_penalty=1.0)

l1_pixel_loss = tf.norm(gan.real_data - gan.generated_data, ord=1)

loss = tf.contrib.gan.losses.combine_adversarial_loss(
    loss, gan, l1_pixel_loss, weight_factor=1)