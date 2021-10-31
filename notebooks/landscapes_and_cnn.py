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

# +
# Import general utilities
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import imshow
# %matplotlib inline

import cv2

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Import TDA utilities
from ripser import Rips
from tadasets import torus, sphere
import persim
import persim.landscapes
from persim.landscapes import PersLandscapeExact

# Import Scikit-Learn tools
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
# -

# Instantiate datasets
data_torus = torus(n=100, c=2, a=1)
data_sphere = sphere(n=100, r=2)

# Instantiate Vietoris-Rips solver
rips = Rips(maxdim = 2)

# Compute persistence diagrams
dgms_torus = rips.fit_transform(data_torus)
dgms_sphere = rips.fit_transform(data_sphere)

# +
# Plot persistence diagrams
fig, axs = plt.subplots(1, 2, dpi=300)
fig.set_size_inches(10, 5)

persim.plot_diagrams(dgms_torus, title="Persistence Diagram of Torus", ax=axs[0])

persim.plot_diagrams(dgms_sphere, title="Persistence Diagram of Sphere", ax=axs[1])

fig.tight_layout()


# +
# Plot persistence landscapes
fig, axs = plt.subplots(1, 2, dpi=300)
fig.set_size_inches(10, 5)

persim.landscapes.plot_landscape_simple(PersLandscapeExact(dgms_torus, 
                                                           hom_deg=1),
                             title="Degree 1 Persistence Landscape of Torus", ax=axs[0])

persim.landscapes.plot_landscape_simple(PersLandscapeExact(dgms_sphere, hom_deg=1),
                            title="Degree 1 Persistence Landscape of Sphere", ax=axs[1])

fig.tight_layout()

# +
# Compute multiple persistence landscapes

landscapes_torus = []
landscapes_sphere = []
nr_torus = 100
nr_sphere = nr_torus

for i in range(nr_torus):
    # Resample data
    _data_torus = torus(n=100, c=2, a=1)
    _data_sphere = sphere(n=100, r=2)

    # Compute persistence diagrams
    dgm_torus = rips.fit_transform(_data_torus)

    # Instantiate persistence landscape transformer
    torus_landscaper = persim.landscapes.PersistenceLandscaper(hom_deg=1, start=0,
                                                               stop=2.0, num_steps=500,
                                                               flatten=False)

    # Compute flattened persistence landscape
    torus_flat = torus_landscaper.fit_transform(dgm_torus)

    landscapes_torus.append(torus_flat)

    # Compute persistence diagrams
    dgm_sphere = rips.fit_transform(_data_sphere)

    # Instantiate persistence landscape transformer
    sphere_landscaper = persim.landscapes.PersistenceLandscaper(hom_deg=1, start=0, stop=2.0,
                                                                num_steps=500, flatten=False)

    # Compute flattened persistence landscape
    sphere_flat = sphere_landscaper.fit_transform(dgm_sphere)

    landscapes_sphere.append(sphere_flat)

print('Torus:', np.shape(landscapes_torus))
print('Sphere:', np.shape(landscapes_sphere))
# -

u = np.max([a.shape[0] for a in landscapes_torus])
v = np.max([a.shape[0] for a in landscapes_sphere])
maximal_length = np.max([u,v])

maximal_length
landscapes_torus[0].shape

# +
# Instantiate zero-padded arrays
padded_torus = np.zeros((nr_torus, maximal_length, 500))
padded_sphere = np.zeros((nr_sphere, maximal_length, 500))

for i, landscape in enumerate(landscapes_torus):
    padded_torus[i, 0:landscape.shape[0], :] = landscape

for i, landscape in enumerate(landscapes_sphere):
    padded_sphere[i, 0:landscape.shape[0], :] = landscape    
# -

landscapes_torus[0].shape

len(landscapes_torus[0])

# +

# padded_torus = landscapes_torus
# padded_sphere = landscapes_sphere
# -

print('Torus:', len(padded_torus))
print('Sphere:', len(padded_sphere))

X = np.concatenate((padded_torus, padded_sphere), axis=0)
y = np.concatenate([np.zeros(nr_torus),np.ones(nr_sphere)])

X.shape

y.shape

# +


X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_tv, y_tv, test_size=0.2)
# -

np.set_printoptions(threshold=np.inf)

# +

plt.figure()

for i, (t, s) in enumerate(zip(padded_torus, padded_sphere)):
    print(i)
    img = Image.fromarray(t, 'L')
    hsize = 2 * img.size[0]
    basewidth = 6 * hsize
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    
    plt.imshow(img)
    plt.show()
    plt.clf() 


    img = Image.fromarray(s, 'L')
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)

    plt.imshow(img)
    plt.show()
    plt.clf()
    print(img.size)
    if i > 5:
        break


# +

model = models.Sequential()
model.add(layers.Conv2D(1, (3, 3), activation='relu', input_shape=(6000, 1000, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(4, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

# -

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_valid, y_valid))


