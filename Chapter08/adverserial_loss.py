import tensorflow as tf

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