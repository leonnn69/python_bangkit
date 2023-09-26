import tensorflow as tf
import numpy as np
from tensorflow import keras

xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0], dtype=float)
ys = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 550.0], dtype=float)

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(xs, ys, epochs=50000)

print(model.predict([100.0]))