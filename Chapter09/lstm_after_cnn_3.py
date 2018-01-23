net = Sequential()
net.add(LSTM(2048,
             return_sequences=False,
             input_shape=input_shape,
             dropout=0.5))
net.add(Dense(512, activation='relu'))
net.add(Dropout(0.5))
net.add(Dense(no_classes, activation='softmax'))