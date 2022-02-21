#!/usr/bin/env python
# coding: utf-8
# %%

# # Cech complexes trial

# %%


import cechmate as cm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

# from tadasets import torus, sphere
import persim
import persim.landscapes
from persim.landscapes import PersLandscapeExact, plot_landscape_simple

# Import ML tools
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import cv2
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models


# %%


df = pd.read_pickle('../data/stats_train.pkl')


# %%


df = df.filter(['modulation_type', 'diagram', 'point_cloud'])
df = df.rename({'diagram': 'diagram'}, axis='columns')

df = df.sample(frac=1)
df = df.sample(n=2000).copy()
df = df.loc[df.modulation_type.isin(['16PSK', '8PSK', '32PSK', 'BPSK'])].copy()


# %%


points_cloud = np.array(df['point_cloud'])


# %%


points_cloud[0]


# %%


mini_cloud=points_cloud[0][0:100]
len(mini_cloud)


# %%


cech = cm.Cech(maxdim=1) #Go up to 1D homology
cech.build(mini_cloud)
dgms_cech0 =  cech.diagrams()


# %%
dgms_cech0[1]

# %%


persim.plot_diagrams(dgms_cech0, title="Persistence Diagram of Cech Complex")


# %%


cech_diagrams = []

for cloud in points_cloud:
    cech.build(cloud[0:100])
    dgms_cech =  cech.diagrams()
    cech_diagrams.append(dgms_cech)


# %%


df['cech_diagrams'] = cech_diagrams
df['cech_shape'] = [len(dgm) for dgm in df.cech_diagrams]


# %%


len(df)


# %%


df = df.loc[df.cech_shape == 2].copy()


# %%


for i, dgm in enumerate(df['cech_diagrams']):
    # print(i)
    if len(dgm) != 2:
        print(i)
        print(len(dgm))


# %%


len(cech_diagrams[305])


# %%





# %%


num_steps=256*2
# Compute multiple persistence landscapes
landscaper = persim.landscapes.PersistenceLandscaper(hom_deg=1,
#                                                      start=0,
#                                                      stop=2.0,
                                                     num_steps=num_steps)

df['landscape'] = [landscaper.fit_transform(dgm) for dgm in df.cech_diagrams]


# %%


df['land_shape'] = [len(land.shape) for land in df.landscape]


# %%


df = df.loc[df.land_shape == 2].copy()


# %%


maximal_length = np.max([a.shape[0] for a in df['landscape']])
maximal_length


# %%


# Instantiate zero-padded arrays
padded = np.zeros((df.shape[0], maximal_length, num_steps))
landscapes = df['landscape'].copy()
for i, landscape in enumerate(landscapes):
    padded[i, 0:landscape.shape[0], :] = landscape


X = np.expand_dims(padded, axis=-1)
y = df.modulation_type.to_numpy()


# %%


from sklearn.preprocessing import LabelBinarizer, LabelEncoder
encoder = LabelBinarizer()
# encoder = LabelEncoder()
y = encoder.fit_transform(y)


# %%


# df = df.sample(n=100)
# test_proportion of 3 means 1/3 so 33% test and 67% train
def shuffle(matrix, target, test_proportion=10):
    ratio = int(matrix.shape[0]/test_proportion)  # should be int
    X_train = matrix[ratio:, :]
    X_test = matrix[:ratio, :]
    Y_train = target[ratio:]
    Y_test = target[:ratio]
    return X_train, X_test, Y_train, Y_test


# %%


X.shape


# %%



# X_train, X_test, Y_train, Y_test = shuffle(X, Y, 3)
X_tv, X_test, y_tv, y_test = shuffle(X, y)
X_train, X_valid, y_train, y_valid = shuffle(X_tv, y_tv)


# %%


X_train.shape


# %%


X_valid.shape


# %%



model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(maximal_length, num_steps, 1)))
model.add(layers.Dropout(0.2))
#model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(y.shape[1], activation="softmax"))


# %%


model.summary()


# %%


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# %%


history = model.fit(X_train, y_train, epochs=15, 
                    batch_size=20,
                    validation_data=(X_valid, y_valid))


# %%


y_pred = model.predict(X_test)
y_p = np.argmax(y_pred, axis=1)
y_t = np.argmax(y_test, axis=1)
cmat = tf.math.confusion_matrix(labels=y_t, predictions=y_p).numpy()
cmat


# %%




