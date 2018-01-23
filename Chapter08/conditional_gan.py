
gan = tf.contrib.gan.gan_model(
    get_generator,
    get_discriminator,
    real_images,
    (input_noise, labels))