from scipy.misc import imsave
import numpy as np
import tensorflow as tf

image_width, image_height = 128, 128
vgg_model = tf.keras.applications.vgg16.VGG16(include_top=False)
input_image = vgg_model.input
vgg_layer_dict = dict([(vgg_layer.name, vgg_layer) for vgg_layer in vgg_model.layers[1:]])
vgg_layer_output = vgg_layer_dict['block5_conv1'].output


filters = []
for filter_idx in range(20):
    loss = tf.keras.backend.mean(vgg_layer_output[:, :, :, filter_idx])
    gradients = tf.keras.backend.gradients(loss, input_image)[0]
    gradient_mean_square = tf.keras.backend.mean(tf.keras.backend.square(gradients))
    gradients /= (tf.keras.backend.sqrt(gradient_mean_square) + 1e-5)
    evaluator = tf.keras.backend.function([input_image], [loss, gradients])

    gradient_ascent_step = 1.
    input_image_data = np.random.random((1, image_width, image_height, 3))
    input_image_data = (input_image_data - 0.5) * 20 + 128

    for i in range(20):
        loss_value, gradient_values = evaluator([input_image_data])
        input_image_data += gradient_values * gradient_ascent_step
        # print('Loss :', loss_value)
        if loss_value <= 0.:
            break

    if loss_value > 0:
        filter = input_image_data[0]
        filter -= filter.mean()
        filter /= (filter.std() + 1e-5)
        filter *= 0.1
        filter += 0.5
        filter = np.clip(filter, 0, 1)
        filter *= 255
        filter = np.clip(filter, 0, 255).astype('uint8')
        filters.append((filter, loss_value))


# For visualisation, not in book

n = 3

filters.sort(key=lambda x: x[1], reverse=True)
filters = filters[:n * n]

margin = 5
width = n * image_width + (n - 1) * margin
height = n * image_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

for i in range(n):
    for j in range(n):
        img, loss = filters[i * n + j]
        stitched_filters[(image_width + margin) * i: (image_width + margin) * i + image_width,
                         (image_height + margin) * j: (image_height + margin) * j + image_height, :] = img

imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
#PIL.Image.fromarray(stitched_filters).save('filters' , 'jpeg')
