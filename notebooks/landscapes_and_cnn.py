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

for i in range(100):
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

# Instantiate zero-padded arrays
ls_torus = np.zeros((100, maximal_length, 500))
ls_sphere = np.zeros((100, maximal_length, 500))

# Populate arrays
for i in range(len(landscapes_torus)):
    ls_torus[i, 0:len(landscapes_torus[i])] = landscapes_torus[i]
    ls_sphere[i, 0:len(landscapes_sphere[i])] = landscapes_sphere[i]
# ls_torus = landscapes_torus
# ls_sphere = landscapes_sphere

print('Torus:', len(ls_torus))
print('Sphere:', len(ls_sphere))

# +
nr_torus = len(ls_torus)
nr_sphere = len(ls_sphere)
X = ls_torus + ls_sphere
y = np.concatenate([np.zeros(nr_torus),np.ones(nr_sphere)])

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_tv, y_tv, test_size=0.2)
# -

np.set_printoptions(threshold=np.inf)
t = ls_torus[0]

# +

plt.figure()

for i, (t, s) in enumerate(zip(ls_torus, ls_sphere)):
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


# +

model = models.Sequential()
model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(6000, 1000, 1),
          strides=(4, 4)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# -

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_valid, y_valid))

# +
# Split points and indicator arrays
P_train, P_test, c_train, c_test = train_test_split(pts, chi, train_size=.8)

# Instantiate support vector classifier
clf = svm.SVC()

# Fit model
clf.fit(P_train, c_train)

# Evaluate model performance using accuracy between ground truth data and predicted data
print(f'Model accuracy: {metrics.accuracy_score(c_test, clf.predict(P_test)):.2f}')

# +
# Delete highly persistent landscapes

# Instantiate trimmed arrays
ls_torus_trim = ls_torus
ls_sphere_trim = ls_sphere

# Trim arrays
for i in range(len(landscapes_torus)):
    ls_torus_trim[i, 0:1000] = np.zeros((1000,))
    ls_sphere_trim[i, 0:1000] = np.zeros((1000,))


print('Torus:', ls_torus_trim.shape)
print('Sphere:', ls_sphere_trim.shape)


# +
# Instantiate PCA solver
pca_torus_trim = PCA(n_components=2)

# Compute PCA
pca_torus_trim.fit_transform(ls_torus_trim)

# Define components
comp_torus_trim = pca_torus_trim.components_

# Instantiate PCA solver
pca_sphere_trim = PCA(n_components=2)

# Compute PCA
pca_sphere_trim.fit_transform(ls_sphere_trim)

# Define components
comp_sphere_trim = pca_sphere_trim.components_

# Plot projection of data onto the first two principal components
plt.figure()
plt.scatter(comp_sphere_trim[0], comp_sphere_trim[1], label='Sphere', alpha=0.4)
plt.scatter(comp_torus_trim[0], comp_torus_trim[1], label='Torus', alpha=0.4)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection with Trimmed Diagrams')
plt.legend()

# +
# Produce lists of points
pts_torus = [[comp_torus[0,i], comp_torus[1,i]] for i in range(len(comp_torus[0]))]
pts_sphere = [[comp_sphere[0,i], comp_sphere[1,i]] for i in range(len(comp_sphere[0]))]

# Instantiate indicator functions
chi_torus = np.zeros(len(pts_torus))
chi_sphere = np.ones(len(pts_sphere))

# Produce final list of points
pts = []

for p in pts_torus:
    pts.append(p)

for p in pts_sphere:
    pts.append(p)

pts = np.array(pts)

# Append indicator functions
chi = np.hstack((chi_torus, chi_sphere))

# +
# Produce lists of points
pts_torus_trim = [[comp_torus_trim[0,i], comp_torus_trim[1,i]] for i in range(len(comp_torus_trim[0]))]
pts_sphere_trim = [[comp_sphere_trim[0,i], comp_sphere_trim[1,i]] for i in range(len(comp_sphere_trim[0]))]

# Instantiate indicator functions
chi_torus_trim = np.zeros(len(pts_torus_trim))
chi_sphere_trim = np.ones(len(pts_sphere_trim))

# Produce final list of points
pts_trim = []

for p in pts_torus_trim:
    pts_trim.append(p)

for p in pts_sphere_trim:
    pts_trim.append(p)

pts_trim = np.array(pts_trim)

# Append indicator functions
chi_trim = np.hstack((chi_torus_trim, chi_sphere_trim))

# Split points and indicator arrays
P_train_trim, P_test_trim, c_train_trim, c_test_trim = train_test_split(pts_trim, chi_trim, train_size=.8)

# Instantiate support vector classifier
clf_trim = svm.SVC()

# Fit model
clf_trim.fit(P_train_trim, c_train_trim)

# Evaluate model performance using accuracy between ground truth data and predicted data
print(f'Model accuracy: {metrics.accuracy_score(c_test_trim, clf_trim.predict(P_test_trim)):.2f}')


# +
# Try multicomponent PCA

# Instantiate multicomponent PCA solver
pca_torus_mcomp = PCA(n_components=6)

# Compute PCA
pca_torus_mcomp.fit_transform(ls_torus)

# Define components
comp_torus_mcomp = pca_torus_mcomp.components_

# Instantiate PCA solver
pca_sphere_mcomp = PCA(n_components=6)

# Compute PCA
pca_sphere_mcomp.fit_transform(ls_sphere)

# Define components
comp_sphere_mcomp = pca_sphere_mcomp.components_


# Produce lists of points
pts_torus_mcomp = [[comp_torus_mcomp[j,i] for j in range(6)] for i in range(len(comp_torus_mcomp[0]))]
pts_sphere_mcomp = [[comp_sphere_mcomp[j,i] for j in range(6)] for i in range(len(comp_sphere_mcomp[0]))]

# Instantiate indicator functions
chi_torus_mcomp = np.zeros(len(pts_torus_mcomp))
chi_sphere_mcomp = np.ones(len(pts_sphere_mcomp))

# Produce final list of points
pts_mcomp = []

for p in pts_torus_mcomp:
    pts_mcomp.append(p)

for p in pts_sphere_mcomp:
    pts_mcomp.append(p)

pts_mcomp = np.array(pts_mcomp)

# Append indicator functions
chi_mcomp = np.hstack((chi_torus_mcomp, chi_sphere_mcomp))

# Split points and indicator arrays
P_train_mcomp, P_test_mcomp, c_train_mcomp, c_test_mcomp = train_test_split(pts_mcomp, chi_mcomp, train_size=.8)

# Instantiate support vector classifier
clf_mcomp = svm.SVC()

# Fit model
clf_mcomp.fit(P_train_mcomp, c_train_mcomp)

# Evaluate model performance using accuracy between ground truth data and predicted data
print(f'Model accuracy: {metrics.accuracy_score(c_test_mcomp, clf_mcomp.predict(P_test_mcomp)):.2f}')

# -


