from os import listdir, mkdir, remove, rename
from os.path import isfile, isdir, join, dirname, basename
import sys
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def list_files(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]


def make_thumb(path):
    subprocess.call(['ffmpeg', '-i', path, '-vf', 'scale=32:32:force_original_aspect_ratio=decrease,pad=32:32:x=(32-iw)/2:y=(32-ih)/2:color=black', '-loglevel', 'panic', join(join(dirname(path), '.thumbs'), basename(path))])
    return path


def make_thumbs(path):
    # get the files
    files = list_files(path)
    # make the thumbs dir if it's not a thing
    thumbs_path = join(path, '.thumbs')
    if not isdir(thumbs_path):
        mkdir(thumbs_path)
    # get the thumbs
    thumbs = list_files(thumbs_path)
    # figure out what needs to be generated
    to_generate = []
    for item in files:
        if not join(join(dirname(item), '.thumbs'), basename(item)) in thumbs:
            to_generate.append(item)
    # start up ffmpeg and wait
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(make_thumb, item) for item in to_generate]
        for item in as_completed(futures):
            pass
    # reset the tty
    subprocess.call(['stty', 'sane'])
    # give back a list of thumbs
    return list_files(thumbs_path)


def get_thumb(path):
    im = Image.open(path, 'r').convert('L')
    width, height = im.size
    data = im.getdata()
    return 1 - (np.array([np.array([data[width * y + x] for x in range(width)]) for y in range(height)]) / 255.0)


def get_set(path):
    likes = make_thumbs(join(path, 'Like'))
    dislikes = make_thumbs(join(path, 'Dislike'))
    labels = []
    images = []
    for item in likes:
        labels.append(1)
        images.append(get_thumb(item))
    for item in dislikes:
        labels.append(0)
        images.append(get_thumb(item))
    return np.array(labels), np.array(images)

labels, images = get_set('Pictures')

"""plt.figure(figsize=(8, 8))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(['Dislike', 'Like'][labels[i]])
plt.show()"""


# setup and train the network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(images, labels, epochs=20)

# get our new images and predict where they should go
old_paths = list_files('Pictures/Like')
old_paths.extend(list_files('Pictures/Dislike'))
old_files = [basename(item) for item in old_paths]
new_paths = make_thumbs('Pictures/New')
new_thumbs = [get_thumb(item) for item in new_paths]
new_paths = [join(dirname(dirname(path)), basename(path)) for path in new_paths]
for i in range(len(new_thumbs)):
    if basename(new_paths[i]) in old_files:
        print('I have this one already: {}'.format(new_paths[i]))
        remove(new_paths[i])
        continue
    dislike, like = model.predict(np.array([new_thumbs[i]]))[0]
    if like > dislike:
        print('like: {}'.format(new_paths[i]))
        rename(new_paths[i], join('Pictures/Like', basename(new_paths[i])))
    else:
        print('dislike: {}'.format(new_paths[i]))
        rename(new_paths[i], join('Pictures/Dislike', basename(new_paths[i])))
shutil.rmtree('Pictures/New/.thumbs')
