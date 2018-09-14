from os import listdir, mkdir, remove, rename
from os.path import isfile, isdir, join, dirname, basename
import sys
import os
import shutil
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def get_image(path):
    im = Image.open(path, 'r').convert('L')
    width, height = im.size
    data = im.getdata()
    return 1 - (np.array([np.array([data[width * y + x] for x in range(width)]) for y in range(height)]) / 255.0)

def get_samples(path, width, height, flatten):
    image = get_image(path)
    image_width = len(image[0])
    image_height = len(image)
    for yoff in range(0, image_height, height):
        for xoff in range(0, image_width, width):
            if flatten:
                yield np.array([image[y+yoff][x+xoff] for x in range(0, width) for y in range(0, height)])
            else:
                yield np.array([np.array([image[y+yoff][x+xoff] for x in range(0, width)]) for y in range(0, height)])

inputs1 = [sample for sample in get_samples('low.png', 16, 16, False)]
inputs2 = [sample for sample in get_samples('med.png', 16, 16, False)]
inputs3 = [sample for sample in get_samples('high.png', 16, 16, False)]
inputs = np.array([*inputs1, *inputs2, *inputs3])
outputs1 = [sample for sample in get_samples('full.png', 16, 16, False)]
outputs2 = [sample for sample in get_samples('full.png', 16, 16, False)]
outputs3 = [sample for sample in get_samples('full.png', 16, 16, False)]
outputs = np.array([*outputs1, *outputs2, *outputs3])

# setup and train the network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(16, 16)),
    keras.layers.Dense(512, activation=tf.nn.sigmoid),
    keras.layers.Dense(512, activation=tf.nn.sigmoid),
    keras.layers.Dense(256, activation=tf.nn.sigmoid),
    keras.layers.Reshape(target_shape=(16, 16))
])
model.compile(tf.train.AdamOptimizer(), loss='mean_squared_error', metrics=['accuracy'])
model.fit(inputs, outputs, epochs=2)
#model.fit(inputs, outputs, epochs=20)

def process(path, new_path, model):
    input_image = get_image(path)
    image_width = len(input_image[0])
    image_height = len(input_image)
    samples = [sample for sample in get_samples(path, 16, 16, False)]
    output_samples = [model.predict(np.array([sample]))[0] for sample in samples]

    x_off = 0
    y_off = 0
    output_buffer = [[0] * image_width] * image_height
    input_buffer = [[0] * image_width] * image_height
    print(len(output_samples))
    print(len(output_samples[0]))
    print(len(output_samples[0][0]))
    for sample in output_samples:
        for y in range(16):
            for x in range(16):
                output_buffer[y_off + y][x_off + x] = sample[y][x]
        x_off += 16
        if x_off >= image_width:
             x_off = 0
             y_off += 16

    x_off = 0
    y_off = 0
    for sample in samples:
        for y in range(16):
            for x in range(16):
                input_buffer[y_off + y][x_off + x] = sample[y][x]
        x_off += 16
        if x_off >= image_width:
             x_off = 0
             y_off += 16


    plt.figure(figsize=(64, 64))
    for i in range(4096):
        plt.subplot(64, 64, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(samples[i], cmap=plt.cm.binary)
    plt.show()

    print(np.array(output_buffer))
    print(np.array(input_buffer))
    output_image = Image.new('L', (image_width, image_height))
    pixels = output_image.load()
    for y in range(image_height):
        for x in range(image_width):
            pixels[y, x] = (int(input_buffer[y][x] * 255),)
    with open(new_path, 'w') as file:
        output_image.save(new_path)

#print(outputs[1000])
#print(model.predict(np.array([inputs[1000]]))[0])
process('med.png', 'test.png', model)
