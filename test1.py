import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((1000, 32))
val_labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))
