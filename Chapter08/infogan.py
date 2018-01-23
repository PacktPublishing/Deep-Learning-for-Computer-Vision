import tensorflow as tf


info_gan = tf.contrib.gan.infogan_model(
    get_generator,
    get_discriminator,
    real_images,
    unstructured_input,
    structured_input)

loss = tf.contrib.gan.gan_loss(
    info_gan,
    gradient_penalty_weight=1,
    gradient_penalty_epsilon=1e-10,
    mutual_information_penalty_weight=1)
