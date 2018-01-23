import numpy as np
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from vgg16_avg import VGG16_Avg
from keras import metrics
from keras.models import Model
from keras import backend as K

import tensorflow as tf

work_dir = ''

content_image = Image.open(work_dir + 'bird_orig.png')

imagenet_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)


def subtract_imagenet_mean(image):
    return (image - imagenet_mean)[:, :, :, ::-1]


def add_imagenet_mean(image, s):
    return np.clip(image.reshape(s)[:, :, :, ::-1] + imagenet_mean, 0, 255)


vgg_model = VGG16_Avg(include_top=False)
content_layer = vgg_model.get_layer('block5_conv1').output
content_model = Model(vgg_model.input, content_layer)

content_image_array = subtract_imagenet_mean(np.expand_dims(np.array(content_image), 0))
content_image_shape = content_image_array.shape
target = K.variable(content_model.predict(content_image_array))


class ConvexOptimiser(object):
    def __init__(self, cost_function, tensor_shape):
        self.cost_function = cost_function
        self.tensor_shape = tensor_shape
        self.gradient_values = None

    def loss(self, point):
        loss_value, self.gradient_values = self.cost_function([point.reshape(self.tensor_shape)])
        return loss_value.astype(np.float64)

    def gradients(self, point):
        return self.gradient_values.flatten().astype(np.float64)


mse_loss = metrics.mean_squared_error(content_layer, target)
grads = K.gradients(mse_loss, vgg_model.input)
cost_function = K.function([vgg_model.input], [mse_loss]+grads)
optimiser = ConvexOptimiser(cost_function, content_image_shape)


def optimise(optimiser, iterations, point, tensor_shape, file_name):
    for i in range(iterations):
        point, min_val, info = fmin_l_bfgs_b(optimiser.loss, point.flatten(),
                                         fprime=optimiser.gradients, maxfun=20)
        point = np.clip(point, -127, 127)
        print('Loss:', min_val)
        imsave(work_dir + 'gen_'+file_name+'_{i}.png', add_imagenet_mean(point.copy(), tensor_shape)[0])
    return point


def generate_rand_img(shape):
    return np.random.uniform(-2.5, 2.5, shape)/1
generated_image = generate_rand_img(content_image_shape)

iterations = 2
generated_image = optimise(optimiser, iterations, generated_image, content_image_shape, 'content')


# Style transfer
style_image = Image.open(work_dir + 'starry_night.png')
style_image = style_image.resize(np.divide(style_image.size, 3.5).astype('int32'))
style_image_array = subtract_imagenet_mean(np.expand_dims(style_image, 0)[:, :, :, :3])
style_image_shape = style_image_array.shape
vgg_model = VGG16_Avg(include_top=False, input_shape=style_image_shape[1:])
style_layers = {layer.name: layer.output for layer in vgg_model.layers}
style_features = [style_layers['block{}_conv1'.format(o)] for o in range(1,3)]
layers_model = Model(vgg_model.input, style_features)
style_targets = [K.variable(feature) for feature in layers_model.predict(style_image_array)]


def grammian_matrix(matrix):
    flattened_matrix = K.batch_flatten(K.permute_dimensions(matrix, (2, 0, 1)))
    matrix_transpose_dot = K.dot(flattened_matrix, K.transpose(flattened_matrix))
    element_count = matrix.get_shape().num_elements()
    return matrix_transpose_dot / element_count


def style_mse_loss(x, y):
    return metrics.mse(grammian_matrix(x), grammian_matrix(y))


style_loss = sum(style_mse_loss(l1[0], l2[0]) for l1, l2 in zip(style_features, style_targets))
grads = K.gradients(style_loss, vgg_model.input)
style_fn = K.function([vgg_model.input], [style_loss]+grads)
optimiser = ConvexOptimiser(style_fn, style_image_shape)

generated_image = generate_rand_img(style_image_shape)
generated_image = optimise(optimiser, iterations, generated_image, style_image_shape, 'style')



w, h = style_image.size
src = content_image_array[:, :h, :w]
outputs = {l.name: l.output for l in vgg_model.layers}

style_layers = [outputs['block{}_conv2'.format(o)] for o in range(1,6)]
content_name = 'block4_conv2'
content_layer = outputs[content_name]

style_model = Model(vgg_model.input, style_layers)
style_targs = [K.variable(o) for o in style_model.predict(style_image_array)]

content_model = Model(vgg_model.input, content_layer)
content_targ = K.variable(content_model.predict(src))

style_wgts = [0.05, 0.2, 0.2, 0.25, 0.3]

loss = sum(style_loss(l1[0], l2[0])*w for l1, l2, w in zip(style_layers, style_targs, style_wgts))
loss += metrics.mse(content_layer, content_targ)/10

grads = tf.keras.backend.gradients(loss, vgg_model.input)
transfer_fn = tf.keras.backend.function([vgg_model.input], [loss]+grads)
evaluator = ConvexOptimiser(transfer_fn, style_image_shape)
enerated_image = generate_rand_img(style_image_shape)
generated_image = optimise(optimiser, iterations, generated_image, style_image_shape)



# Content plus style transfer


style_width, style_height = style_image.size
content_image_array = content_image_array[:, :style_height, :style_width]

style_layers_2 = [style_layers['block{}_conv2'.format(block_no)] for block_no in range(1,6)]
content_layer = style_layers['block4_conv2']

style_model = Model(vgg_model.input, style_layers_2)
style_targets = [K.variable(o) for o in style_model.predict(style_image_array)]

content_model = Model(vgg_model.input, content_layer)
content_target = K.variable(content_model.predict(content_image_array))

style_weights = [0.05, 0.2, 0.2, 0.25, 0.3]

style_loss = sum(style_loss(l1[0], l2[0])*w for l1, l2, w in zip(style_layers_2, style_targets, style_weights))
content_loss = metrics.mse(content_layer, content_target)/10
loss = style_loss + content_loss

gradients = K.gradients(loss, vgg_model.input)
transfer_fn = K.function([vgg_model.input], [loss]+gradients)
optimiser = ConvexOptimiser(transfer_fn, style_image_shape)
generated_image = generate_rand_img(style_image_shape)
generated_image = optimise(optimiser, iterations, generated_image, style_image_shape)
