import image_preprocessor
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras

if __name__ == '__main__':
    input_data = image_preprocessor.generate_data()

samples = input_data.shape[0]
vec_len = input_data.shape[1]


model = keras.Sequential([
    keras.layers.Dense(vec_len, input_dim=vec_len),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(192, activation=tf.nn.relu),  # bottleneck stage
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(vec_len, activation=tf.nn.relu)
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())
model.fit(input_data, input_data, verbose=1, epochs=10, batch_size=256)

compressor = keras.Sequential([
    keras.layers.Dense(vec_len, input_dim=vec_len, weights=model.layers[0].get_weights()),
    keras.layers.Dense(256, activation=tf.nn.relu, weights=model.layers[1].get_weights()),
    keras.layers.Dense(192, activation=tf.nn.relu, weights=model.layers[2].get_weights())
])

compressor.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())

test_out = compressor.predict(input_data[172].reshape(1, vec_len))  # should be 128x1 output
print(test_out)

test = input_data[172].reshape(1, vec_len)  # random sample
y_test = model.predict(test)

test = np.reshape(test, (-1, 28))  # change 28 to image dims
y_test = np.reshape(y_test, (-1, 28))  # change 28 to image dims

cv2.imshow('Test Image', test)
cv2.imshow('Output Image', y_test)
cv2.waitKey(0)
