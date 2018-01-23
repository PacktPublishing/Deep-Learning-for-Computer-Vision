####################
#gan estimator

tf.reset_default_graph()

def _get_train_input_fn(batch_size, noise_dims):
    def train_input_fn():
        with tf.device('/cpu:0'):
            images, _, _ = data_provider.provide_data(
                'train', batch_size, MNIST_DATA_DIR)
        noise = tf.random_normal([batch_size, noise_dims])
        return noise, images
    return train_input_fn


def _get_predict_input_fn(batch_size, noise_dims):
    def predict_input_fn():
        noise = tf.random_normal([batch_size, noise_dims])
        return noise
    return predict_input_fn

BATCH_SIZE = 32
NOISE_DIMS = 64
NUM_STEPS = 2000

# Initialize GANEstimator with options and hyperparameters.
gan_estimator = tfgan.estimator.GANEstimator(
    generator_fn=generator_fn,  # Same as before.
    discriminator_fn=discriminator_fn,  # Same as before.
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
    add_summaries=tfgan.estimator.SummaryType.IMAGES)

# Train estimator.
train_input_fn = _get_train_input_fn(BATCH_SIZE, NOISE_DIMS)
gan_estimator.train(train_input_fn, max_steps=NUM_STEPS)

# Run inference.
predict_input_fn = _get_predict_input_fn(36, NOISE_DIMS)
prediction_iterable = gan_estimator.predict(predict_input_fn)
predictions = [prediction_iterable.next() for _ in xrange(36)]

# Nicely tile output and visualize.
image_rows = [np.concatenate(predictions[i:i+6], axis=0) for i in
              range(0, 36, 6)]
tiled_images = np.concatenate(image_rows, axis=1)

# Visualize.
plt.axis('off')
plt.imshow(np.squeeze(tiled_images), cmap='gray')