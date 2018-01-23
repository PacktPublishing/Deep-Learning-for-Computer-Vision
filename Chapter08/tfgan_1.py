import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

def provide_data(split_name, batch_size, dataset_dir, num_readers=1,
                 num_threads=1):
  """Provides batches of MNIST digits.

  Args:
    split_name: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    dataset_dir: The directory where the MNIST data can be found.
    num_readers: Number of dataset readers.
    num_threads: Number of prefetching threads.

  Returns:
    images: A `Tensor` of size [batch_size, 28, 28, 1]
    one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
      each row has a single element set to one and the rest set to zeros.
    num_samples: The number of total samples in the dataset.

  Raises:
    ValueError: If `split_name` is not either 'train' or 'test'.
  """
  dataset = datasets.get_dataset('mnist', split_name, dataset_dir=dataset_dir)
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size,
      shuffle=(split_name == 'train'))
  [image, label] = provider.get(['image', 'label'])

  # Preprocess the images.
  image = (tf.to_float(image) - 128.0) / 128.0

  # Creates a QueueRunner for the pre-fetching operation.
  images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=5 * batch_size)

  one_hot_labels = tf.one_hot(labels, dataset.num_classes)
  return images, one_hot_labels, dataset.num_samples


def float_image_to_uint8(image):

  image = (image * 128.0) + 128.0
  return tf.cast(image, tf.uint8)
#images = provide_data(10000)
latent_dimensions = 64

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
    #generator = tf.layers.batch_normalization(generator, training=is_training)
    generator = dense_layer(generator, 7 * 7 * 256)
    #generator = tf.layers.batch_normalization(generator, training=is_training)
    generator = tf.reshape(generator,  [-1, 7, 7, 256])
    generator = transpose_convolution_layer(generator, 64)
    #generator = tf.layers.batch_normalization(generator, training=is_training)
    generator = transpose_convolution_layer(generator, 32)

    print(generator)
    #generator = tf.layers.batch_normalization(generator, training=is_training)
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
#    discriminator = tf.layers.batch_normalization(discriminator, training=is_training)
    discriminator = dense_layer(discriminator, 2)
    return discriminator

test_images, test_labels = mnist_data.test.images, mnist_data.test.labels
x_input = tf.placeholder(tf.float32, shape=[None, 784])
x_input_reshape = tf.reshape(x_input, [-1, 28, 28, 1],
                             name='input_reshape')
session = tf.Session()
images = session.run(x_input_reshape, feed_dict={x_input:test_images})
session.close()

random_images = tf.random_normal([10000, latent_dimensions])
gan = tf.contrib.gan.gan_model(
    get_generator,
    get_discriminator,
    images,
    random_images
)
gan_loss = tf.contrib.gan.gan_loss(
    gan,
    generator_loss_fn=tf.contrib.gan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tf.contrib.gan.losses.wasserstein_discriminator_loss)

train_ops = tf.contrib.gan.gan_train_ops(
    gan,
    gan_loss,
    generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5))

tf.contrib.gan.gan_train(
    train_ops,
    hooks=[tf.train.StopAtStepHook(num_steps=10)],
    logdir='/tmp/gan')



#################################
#Conditioanl gan

tf.reset_default_graph()

# Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
# for forward and backwards propogation.
batch_size = 32
with tf.device('/cpu:0'):
    images, one_hot_labels, _ = data_provider.provide_data('train', batch_size, MNIST_DATA_DIR)

# Sanity check that we're getting images.
imgs_to_visualize = tfgan.eval.image_reshaper(images[:20,...], num_cols=10)
visualize_digits(imgs_to_visualize)


def conditional_generator_fn(inputs, weight_decay=2.5e-5):
    """Generator to produce MNIST images.

    Args:
        inputs: A 2-tuple of Tensors (noise, one_hot_labels).
        weight_decay: The value of the l2 weight decay.

    Returns:
        A generated image in the range [-1, 1].
    """
    noise, one_hot_labels = inputs

    with slim.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(noise, 1024)
        net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

        return net


def conditional_discriminator_fn(img, conditioning, weight_decay=2.5e-5):
    """Conditional discriminator network on MNIST digits.

    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        conditioning: A 2-tuple of Tensors representing (noise, one_hot_labels).
        weight_decay: The L2 weight decay.

    Returns:
        Logits for the probability that the image is real.
    """
    _, one_hot_labels = conditioning
    with slim.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)

        return layers.linear(net, 1)

noise_dims = 64
conditional_gan_model = tfgan.gan_model(
    generator_fn=conditional_generator_fn,
    discriminator_fn=conditional_discriminator_fn,
    real_data=images,
    generator_inputs=(tf.random_normal([batch_size, noise_dims]),
                      one_hot_labels))

# Sanity check that currently generated images are garbage.
cond_generated_data_to_visualize = tfgan.eval.image_reshaper(
    conditional_gan_model.generated_data[:20,...], num_cols=10)
visualize_digits(cond_generated_data_to_visualize)

gan_loss = tfgan.gan_loss(
    conditional_gan_model, gradient_penalty_weight=1.0)

# Sanity check that we can evaluate our losses.
evaluate_tfgan_loss(gan_loss)

generator_optimizer = tf.train.AdamOptimizer(0.0009, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    conditional_gan_model,
    gan_loss,
    generator_optimizer,
    discriminator_optimizer)

# Set up class-conditional visualization. We feed class labels to the generator
# so that the the first column is `0`, the second column is `1`, etc.
images_to_eval = 500
assert images_to_eval % 10 == 0

random_noise = tf.random_normal([images_to_eval, 64])
one_hot_labels = tf.one_hot(
    [i for _ in xrange(images_to_eval // 10) for i in xrange(10)], depth=10)
with tf.variable_scope(conditional_gan_model.generator_scope, reuse=True):
    eval_images = conditional_gan_model.generator_fn(
        (random_noise, one_hot_labels))
reshaped_eval_imgs = tfgan.eval.image_reshaper(
    eval_images[:20, ...], num_cols=10)

# We will use a pretrained classifier to measure the progress of our generator.
# Specifically, the cross-entropy loss between the generated image and the target
# label will be the metric we track.
MNIST_CLASSIFIER_FROZEN_GRAPH = './mnist/data/classify_mnist_graph_def.pb'
xent_score = util.mnist_cross_entropy(
    eval_images, one_hot_labels, MNIST_CLASSIFIER_FROZEN_GRAPH)


global_step = tf.train.get_or_create_global_step()
train_step_fn = tfgan.get_sequential_train_steps()
loss_values, xent_score_values  = [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with slim.queues.QueueRunners(sess):
        start_time = time.time()
        for i in xrange(2001):
            cur_loss, _ = train_step_fn(
                sess, gan_train_ops, global_step, train_step_kwargs={})
            loss_values.append((i, cur_loss))
            if i % 50 == 0:
                xent_score_values.append((i, sess.run(xent_score)))
            if i % 400 == 0:
                print('Current loss: %f' % cur_loss)
                print('Current cross entropy score: %f' % xent_score_values[-1][1])
                visualize_training_generator(i, start_time, sess.run(reshaped_eval_imgs))

# Plot the eval metrics over time.
plt.title('Cross entropy score per step')
plt.plot(*zip(*xent_score_values))
plt.figure()
plt.title('Training loss per step')
plt.plot(*zip(*loss_values))
plt.show()


###################
# Info gan

tf.reset_default_graph()

# Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
# for forward and backwards propogation.
batch_size = 32
with tf.device('/cpu:0'):
    images, _, _ = data_provider.provide_data('train', batch_size, MNIST_DATA_DIR)

# Sanity check that we're getting images.
imgs_to_visualize = tfgan.eval.image_reshaper(images[:20,...], num_cols=10)
visualize_digits(imgs_to_visualize)


def infogan_generator(inputs, categorical_dim, weight_decay=2.5e-5):
    """InfoGAN discriminator network on MNIST digits.

    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.

    Args:
        inputs: A 3-tuple of Tensors (unstructured_noise, categorical structured
            noise, continuous structured noise). `inputs[0]` and `inputs[2]` must be
            2D, and `inputs[1]` must be 1D. All must have the same first dimension.
        categorical_dim: Dimensions of the incompressible categorical noise.
        weight_decay: The value of the l2 weight decay.

    Returns:
        A generated image in the range [-1, 1].
    """
    unstructured_noise, cat_noise, cont_noise = inputs
    cat_noise_onehot = tf.one_hot(cat_noise, categorical_dim)
    all_noise = tf.concat([unstructured_noise, cat_noise_onehot, cont_noise], axis=1)

    with slim.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(all_noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

        return net


def infogan_discriminator(img, unused_conditioning, weight_decay=2.5e-5,
                          categorical_dim=10, continuous_dim=2):
    """InfoGAN discriminator network on MNIST digits.

    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.

    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
        categorical_dim: Dimensions of the incompressible categorical noise.
        continuous_dim: Dimensions of the incompressible continuous noise.

    Returns:
        Logits for the probability that the image is real, and a list of posterior
        distributions for each of the noise vectors.
    """
    with slim.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

        logits_real = layers.fully_connected(net, 1, activation_fn=None)

        # Recognition network for latent variables has an additional layer
        encoder = layers.fully_connected(net, 128, normalizer_fn=layers.batch_norm)

        # Compute logits for each category of categorical latent.
        logits_cat = layers.fully_connected(
            encoder, categorical_dim, activation_fn=None)
        q_cat = ds.Categorical(logits_cat)

        # Compute mean for Gaussian posterior of continuous latents.
        mu_cont = layers.fully_connected(
            encoder, continuous_dim, activation_fn=None)
        sigma_cont = tf.ones_like(mu_cont)
        q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)

        return logits_real, [q_cat, q_cont]


# Dimensions of the structured and unstructured noise dimensions.
cat_dim, cont_dim, noise_dims = 10, 2, 64

generator_fn = functools.partial(infogan_generator, categorical_dim=cat_dim)
discriminator_fn = functools.partial(
    infogan_discriminator, categorical_dim=cat_dim,
    continuous_dim=cont_dim)
unstructured_inputs, structured_inputs = util.get_infogan_noise(
    batch_size, cat_dim, cont_dim, noise_dims)

infogan_model = tfgan.infogan_model(
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    real_data=images,
    unstructured_generator_inputs=unstructured_inputs,
    structured_generator_inputs=structured_inputs)

infogan_loss = tfgan.gan_loss(
    infogan_model,
    gradient_penalty_weight=1.0,
    mutual_information_penalty_weight=1.0)

# Sanity check that we can evaluate our losses.
evaluate_tfgan_loss(infogan_loss)


generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    infogan_model,
    infogan_loss,
    generator_optimizer,
    discriminator_optimizer)

# Set up images to evaluate MNIST score.
images_to_eval = 500
assert images_to_eval % cat_dim == 0

unstructured_inputs = tf.random_normal([images_to_eval, noise_dims-cont_dim])
cat_noise = tf.constant(range(cat_dim) * (images_to_eval // cat_dim))
cont_noise = tf.random_uniform([images_to_eval, cont_dim], -1.0, 1.0)

with tf.variable_scope(infogan_model.generator_scope, reuse=True):
    eval_images = infogan_model.generator_fn(
        (unstructured_inputs, cat_noise, cont_noise))

MNIST_CLASSIFIER_FROZEN_GRAPH = './mnist/data/classify_mnist_graph_def.pb'
eval_score = util.mnist_score(
    eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)

# Generate three sets of images to visualize the effect of each of the structured noise
# variables on the output.
rows = 2
categorical_sample_points = np.arange(0, 10)
continuous_sample_points = np.linspace(-1.0, 1.0, 10)
noise_args = (rows, categorical_sample_points, continuous_sample_points,
              noise_dims-cont_dim, cont_dim)

display_noises = []
display_noises.append(util.get_eval_noise_categorical(*noise_args))
display_noises.append(util.get_eval_noise_continuous_dim1(*noise_args))
display_noises.append(util.get_eval_noise_continuous_dim2(*noise_args))

display_images = []
for noise in display_noises:
    with tf.variable_scope(infogan_model.generator_scope, reuse=True):
        display_images.append(infogan_model.generator_fn(noise))

display_img = tfgan.eval.image_reshaper(
    tf.concat(display_images, 0), num_cols=10)

global_step = tf.train.get_or_create_global_step()
train_step_fn = tfgan.get_sequential_train_steps()
loss_values, mnist_score_values  = [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with slim.queues.QueueRunners(sess):
        start_time = time.time()
        for i in xrange(6001):
            cur_loss, _ = train_step_fn(
            sess, gan_train_ops, global_step, train_step_kwargs={})
            loss_values.append((i, cur_loss))
            if i % 100 == 0:
                mnist_score_values.append((i, sess.run(eval_score)))
            if i % 1500 == 0:
                print('Current loss: %f' % cur_loss)
                print('Current MNIST score: %f' % mnist_score_values[-1][1])
                visualize_training_generator(i, start_time, sess.run(display_img))


# Plot the eval metrics over time.
plt.title('MNIST Score per step')
plt.plot(*zip(*mnist_score_values))
plt.figure()
plt.title('Training loss per step')
plt.plot(*zip(*loss_values))


# ADAM variables are causing the checkpoint reload to choke, so omit them when
# doing variable remapping.
var_dict = {x.op.name: x for x in
            tf.contrib.framework.get_variables('Generator/')
            if 'Adam' not in x.name}
tf.contrib.framework.init_from_checkpoint(
    './mnist/data/infogan_model.ckpt',
    var_dict)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    display_img_np = sess.run(display_img)
plt.axis('off')
plt.imshow(np.squeeze(display_img_np), cmap='gray')
plt.show()

