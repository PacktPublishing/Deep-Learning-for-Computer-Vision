import tensorflow as tf

gan = tf.contrib.gan.gan_model(
    get_generator,
    get_discriminator,
    real_images,
    input_images)

loss = tf.contrib.gan.gan_loss(
    gan,
    tf.contrib.gan.losses.least_squares_generator_loss,
    tf.contrib.gan.losses.least_squares_discriminator_loss)

l1_loss = tf.norm(gan.real_data - gan.generated_data, ord=1)

gan_loss = tf.contrib.gan.losses.combine_adversarial_loss(
    loss, gan, l1_loss, weight_factor=1)
