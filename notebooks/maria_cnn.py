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

df = pd.read_pickle('../data/stats_train.pkl')
df = df.sample(n=100)
df.columns

# Instantiate datasets
num_steps = 1000
num_points = 100
# Instantiate Vietoris-Rips solver
rips = Rips(maxdim=2)

# +
# Compute multiple persistence landscapes
landscaper = persim.landscapes.PersistenceLandscaper(hom_deg=1,
                                                     start=0,
                                                     # stop=1.0,
                                                     num_steps=num_steps,
                                                     flatten=False)

df['landscape'] = [landscaper.fit_transform(dgm) for dgm in df.diagram]

# -

type(df.iloc[0].landscape)

maximal_length = np.max([a.shape[0] for a in df['landscape']])
maximal_length

land = np.expand_dims(df.iloc[0].landscape, axis=-1)
land.shape

df['padded'] = [tf.image.pad_to_bounding_box(
               tf. convert_to_tensor(np.expand_dims(land, axis=-1)),
                                     maximal_length - land.shape[0],
                                     0,
                                     maximal_length, 1000)
                for land in df['landscape']]

# df = df.sample(n=100)
df.shape

X = df.padded
y = df.modulation_id

X.iloc[4].shape

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_tv, y_tv,
                                                      test_size=0.5)

X_train.shape

# +

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu',
                        input_shape=(maximal_length, num_steps, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation="softmax"))

# -

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, 
                    validation_data=(X_valid, y_valid))

y_valid


