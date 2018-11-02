import tensorflow as tf

def simple_cnn(input_shape, no_classes):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Dense(units=no_classes, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
                  
    return model