# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
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
import matplotlib.pyplot as plt

# Import TDA utilities
from ripser import Rips
import persim.landscapes

# Import ML tools
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models
# -

# %autoreload 2
# %matplotlib inline

# # Path

# +
df = pd.read_pickle('../data/stats_train.pkl')

# df = pd.read_pickle('../data/stats_size_10010.pkl')
# -

df.columns


def shuffle(matrix, target, test_proportion=10):
    # applicable only if data are randomly sorted
    # test_proportion of 3 means 1/3 so 33% test and 67% train
    ratio = int(matrix.shape[0]/test_proportion)  # should be int
    X_train = matrix[ratio:, :]
    X_test = matrix[:ratio, :]
    Y_train = target[ratio:]
    Y_test = target[:ratio]
    return X_train, X_test, Y_train, Y_test


df.columns

# +
df = df.filter(['modulation_type', 'diagram'])
df = df.rename({'diagram': 'diagram'}, axis='columns')


df = df.sample(n=2000).copy()
df = df.loc[df.modulation_type.isin(['16PSK', '8PSK', '32PSK', 'BPSK'])].copy()
# df = df.loc[df.modulation_type.isin(['32PSK', 'QPSK', 'BPSK'])].copy()
# -

df = df.sample(frac=1)  # do not comment this line! 
df.modulation_type.unique()

# Instantiate datasets
num_steps = 256
# Instantiate Vietoris-Rips solver
rips = Rips(maxdim=2)

# #### Consider ever 
# #### 1) standarazeise persistent diagrams or clouds and set start and stop or
# #### 2) adjust start and stop and keep original scale of homology diagrams, 
# #### then values of lambda functions could also contain some information

# +
# Compute multiple persistence landscapes
# 
landscaper = persim.landscapes.PersistenceLandscaper(hom_deg=1,
#                                                      start=0,
#                                                      stop=2.0,
                                                     num_steps=num_steps)

df['landscape'] = [landscaper.fit_transform(dgm) for dgm in df.diagram]
# -

# #### landscaper.fit_transform(dgm) returns a table with values (not cooridnates!) of lambda functions for x in range(start, stop) and step determined by num_steps. For each lambda function there exist one row in this table. 

df['landscape'].iloc[0][0]

df.shape

df['land_shape'] = [len(land.shape) for land in df.landscape]

# drop empty landscapes
df = df.loc[df.land_shape == 2].copy()
df = df.sample(frac=1)

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
# y

# +

encoder = LabelBinarizer()
# encoder = LabelEncoder()
y = encoder.fit_transform(y)
# -

y.shape

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
# model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(y.shape[1], activation="softmax"))

# -

y_valid

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=6, 
                    batch_size=20,
                    validation_data=(X_valid, y_valid))

y_pred = model.predict(X_test)

y_p = np.argmax(y_pred, axis=1)
y_p

y_t = np.argmax(y_test, axis=1)
y_t

cm = tf.math.confusion_matrix(labels=y_t, predictions=y_p).numpy()

cm


