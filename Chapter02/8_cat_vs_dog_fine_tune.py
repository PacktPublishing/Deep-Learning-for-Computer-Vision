import tensorflow as tf
import os

work_dir = ''

weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'

image_height, image_width = 150, 150
train_dir = os.path.join(work_dir, 'train')
test_dir = os.path.join(work_dir, 'test')
no_classes = 2
no_validation = 800
epochs = 50
batch_size = 32
no_train = 2000
no_test = 800
input_shape = (image_height, image_width, 3)
epoch_steps = no_train // batch_size
test_steps = no_test // batch_size

model = tf.keras.applications.VGG16(include_top=False)
model_fine_tune = tf.keras.models.Sequential()
model_fine_tune.add(tf.keras.layers.Flatten(input_shape=model.output_shape))
model_fine_tune.add(tf.keras.layers.Dense(256, activation='relu'))
model_fine_tune.add(tf.keras.layers.Dropout(0.5))
model_fine_tune.add(tf.keras.layers.Dense(no_classes, activation='softmax'))

model_fine_tune.load_weights(top_model_weights_path)
model.add(model_fine_tune)

for vgg_layer in model.layers[:25]:
    vgg_layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

generator_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    zoom_range=0.3,
    shear_range=0.3
    )

generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

generator_train = generator_train.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=(image_width, image_height)
)

generator_test = generator_test.flow_from_directory(
    test_dir,
    batch_size=batch_size,
    target_size=(image_width, image_height)
)

model.fit_generator(
    generator_train,
    steps_per_epoch=epoch_steps,
    epochs=epochs,
    validation_data=generator_test,
    validation_steps=test_steps
)