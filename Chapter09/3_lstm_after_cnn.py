import tensorflow as tf
input_shape = [500,500]
no_classes = 2

net = tf.keras.models.Sequential()
net.add(tf.keras.layers.LSTM(2048,
             return_sequences=False,
             input_shape=input_shape,
             dropout=0.5))
net.add(tf.keras.layers.Dense(512, activation='relu'))
net.add(tf.keras.layers.Dropout(0.5))
net.add(tf.keras.layers.Dense(no_classes, activation='softmax'))