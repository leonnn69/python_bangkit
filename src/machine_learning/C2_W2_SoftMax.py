import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging

def my_softmax(z):
    ez = np.exp(z)
    sm = ez / np.sum(ez)
    return sm

plt.close("all")
plt_softmax(my_softmax)

# Tensorflow
# This lab will discuss two ways of implementing the softmax, cross-entropy loss in Tensorflow, 
# the 'obvious' method and the 'preferred' method. 
# The former is the most straightforward while the latter is more numerically stable.

# Let's start by creating a dataset to train a multiclass classification model.
# data sets
centers = [[-5, 2], [-2,-2], [1, 2], [5,-2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)

model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)

# Because the softmax is integrated into the output layer, the output is a vector of probabilities.
p_nonpreferred = model.predict(X_train)
print(p_nonpreferred [:2])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))

# preferred
preferred_model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')   #<-- Note
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=10
)

p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:2]}")
print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))

# The output predictions are not probabilities! If the desired output are probabilities, the output should be be processed by a softmax.

sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred)) 

# To select the most likely category, the softmax is not required. 
# One can find the index of the largest output using np.argmax().

for i in range(5):
    print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")