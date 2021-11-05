# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext pycodestyle_magic
# %flake8_on --ignore E703,E702
# %load_ext autoreload

# +
# Import general utilities
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import imshow

# Import TDA utilities
from ripser import Rips
from tadasets import torus, sphere
import persim
import persim.landscapes
from persim.landscapes import PersLandscapeExact, plot_landscape_simple
import path
import modurec as mr

# Import ML tools
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# -

# %autoreload 2
# %matplotlib inline

# +
# df = pd.read_pickle('../data/stats_train.pkl')

df = pd.read_pickle('../data/stats_train.pkl')
df = df.filter(['modulation_type', 'diagram'])
# df = df.sample(n=2000).copy()
df = df.loc[df.modulation_type.isin(['16PSK', '8PSK', '32PSK', 'BPSK'])].copy()
# df = df.loc[df.modulation_type.isin(['32PSK', 'QPSK', 'BPSK'])].copy()
# -

df.modulation_type.unique()

# Instantiate datasets
num_steps = 258
# Instantiate Vietoris-Rips solver
rips = Rips(maxdim=2)

# +
# Compute multiple persistence landscapes
landscaper = persim.landscapes.PersistenceLandscaper(hom_deg=1,
#                                                      start=0,
#                                                      stop=2.0,
                                                     num_steps=num_steps)

df['landscape'] = [landscaper.fit_transform(dgm) for dgm in df.diagram]

# -

df['land_shape'] = [len(land.shape) for land in df.landscape]

df = df.loc[df.land_shape == 2].copy()

maximal_length = np.max([a.shape[0] for a in df['landscape']])
maximal_length

# +
# Instantiate zero-padded arrays
padded = np.zeros((df.shape[0], maximal_length, num_steps))
landscapes = df['landscape'].copy()
for i, landscape in enumerate(landscapes):
    padded[i, 0:landscape.shape[0], :] = landscape


X = np.expand_dims(padded, axis=-1)
y = df.modulation_type.to_numpy()
# -

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y = encoder.fit_transform(y)


# df = df.sample(n=100)
# test_proportion of 3 means 1/3 so 33% test and 67% train
def shuffle(matrix, target, test_proportion=10):
    ratio = int(matrix.shape[0]/test_proportion)  # should be int
    X_train = matrix[ratio:, :]
    X_test = matrix[:ratio, :]
    Y_train = target[ratio:, :]
    Y_test = target[:ratio, :]
    return X_train, X_test, Y_train, Y_test


X.shape

# +


# X_train, X_test, Y_train, Y_test = shuffle(X, Y, 3)
X_tv, X_test, y_tv, y_test = shuffle(X, y)
X_train, X_valid, y_train, y_valid = shuffle(X_tv, y_tv)
# -

X_train.shape

X_valid.shape

# +

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(maximal_length, num_steps, 1)))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(y.shape[1], activation="softmax"))

# -

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    batch_size=20,
                    validation_data=(X_valid, y_valid))

y_valid


