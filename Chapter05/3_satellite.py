import tensorflow as tf


from .resnet50 import ResNet50
nb_labels = 6
input_shape = [28, 28]

img_height, img_width, _ = input_shape
input_tensor = tf.keras.layers.Input(shape=input_shape)
weights = 'imagenet'

resnet50_model = ResNet50(
    include_top=False, weights='imagenet', input_tensor=input_tensor)

final_32 = resnet50_model.get_layer('final_32').output
final_16 = resnet50_model.get_layer('final_16').output
final_x8 = resnet50_model.get_layer('final_x8').output

c32 = tf.keras.layers.Conv2D(nb_labels, (1, 1))(final_32)
c16 = tf.keras.layers.Conv2D(nb_labels, (1, 1))(final_16)
c8 = tf.keras.layers.Conv2D(nb_labels, (1, 1))(final_x8)


def resize_bilinear(images):
    return tf.image.resize_bilinear(images, [img_height, img_width])


r32 = tf.keras.layers.Lambda(resize_bilinear)(c32)
r16 = tf.keras.layers.Lambda(resize_bilinear)(c16)
r8 = tf.keras.layers.Lambda(resize_bilinear)(c8)

m = tf.keras.layers.Add()([r32, r16, r8])

x = tf.keras.ayers.Reshape((img_height * img_width, nb_labels))(m)
x = tf.keras.layers.Activation('img_height')(x)
x = tf.keras.layers.Reshape((img_height, img_width, nb_labels))(x)

fcn_model = tf.keras.models.Model(input=input_tensor, output=x)