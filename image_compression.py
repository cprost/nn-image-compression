import image_preprocessor
import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow import keras

'''
Run this file to download the MNIST/CIFAR10 dataset(s) and prepare them for the neural network.

The neural network architecture is specified in 'model', and takes 'vec_len' and 'bottleneck'
as variable parameters. The 'bottleneck' value determines the degree of compression in the
autoencoder.

Predicting any sample can be tested with 'predict_output(<SAMPLE #>)'
'''

def image_grid(img_data, indices, suffix):
    dims = img_data[0].shape

    grid = np.ones((dims[0], 10))
    grid = cv2.normalize(grid, None, 255, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    buffer = grid.copy()

    for i in indices:
        current = cv2.normalize(img_data[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        grid = np.concatenate((grid, current), axis=1)
        grid = np.concatenate((grid, buffer), axis=1)

    currdate = time.strftime("%Y%m%d_%H%M%S")
    filename = 'autoencoder_' + currdate + "_" + suffix + '.png'

    grid = grid.astype('uint8')
    cv2.imwrite(filename, grid)

def unflatten(img_data, dims):
    samples = len(img_data)
    new_data = img_data.reshape(samples, dims[1], dims[0]) # vector to original image size
    return new_data

def predict_output(idx):
    input = input_data[idx]
    input = input.reshape(1, vec_len) # vector to original image size
    output = model.predict(input)
    output = output.reshape(dims[1], dims[0])
    input = input.reshape(dims[1], dims[0])
    cv2.imshow('Original Image at Index = ' + str(idx), input)
    cv2.imshow('Reproduced Image at Index = ' + str(idx), output)
    cv2.waitKey(0)

def sample_output(idx):
    encoded_out = compressor.predict(input_data[idx].reshape(1, vec_len)) # vector to original image size
    decoded_out = decoder.predict(encoded_out)
    decoded_out = decoded_out.reshape(dims[1], dims[0])
    cv2.imshow('Sample of separated encoder-decoder results', decoded_out)
    cv2.waitKey(0)

# NEURAL NETWORK BEGINS BELOW

if __name__ == '__main__':
    input_data, samples, dims = image_preprocessor.generate_data()

# saved input_data can also be imported using 'input_data = np.load('cleaned_<DATASET>_data.npy')

vec_len = dims[0] * dims[1]
bottleneck = 512  # size of the bottleneck layer in neurons

model = keras.Sequential([
    keras.layers.Dense(vec_len, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(bottleneck, activation=tf.nn.relu),  # bottleneck stage
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(vec_len, activation=tf.nn.relu)
])

model.compile(optimizer='adam', loss='mse')
model.fit(input_data, input_data, verbose=1, epochs=1, batch_size=256)

compressor = keras.Sequential([
    keras.layers.Dense(vec_len, activation=tf.nn.relu, weights=model.layers[0].get_weights()),
    keras.layers.Dense(256, activation=tf.nn.relu, weights=model.layers[1].get_weights()),
    keras.layers.Dense(bottleneck, activation=tf.nn.relu, weights=model.layers[2].get_weights())
])

decoder = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu, input_dim=bottleneck, weights=model.layers[3].get_weights()),
    keras.layers.Dense(vec_len, activation=tf.nn.relu, weights=model.layers[4].get_weights())
])

mytest = compressor.predict(input_data[5].reshape(1, vec_len))
myout = decoder.predict(mytest)
myout = myout.reshape(dims[1], dims[0])
cv2.imshow('el', myout)
cv2.waitKey(0)

compressor.compile(optimizer='adam', loss='mse')
decoder.compile(optimizer='adam', loss='mse')

sample_idx = [283, 277, 272, 299, 295] # sample indices of even numbers in MNIST, otherwise random
sample_in = input_data[sample_idx]
sample_in = sample_in.reshape(len(sample_idx), 1, vec_len)

output_data = model.predict(sample_in[0])
for i in range(1, len(sample_idx)):
    output_data = np.append(output_data, model.predict(sample_in[i]), axis=0)

output_data = output_data.reshape(len(sample_idx), 1, vec_len)

# restores flattened vectors to original image shape
restored_input = unflatten(input_data, dims)
restored_output = unflatten(output_data, dims)

# prints image grids to file
image_grid(restored_input, sample_idx, 'in')
image_grid(restored_output, np.arange(len(sample_idx)), 'out')

# sample prediction at mysample = <SAMPLE #> in dataset
# mysample = 999
# predict_output(mysample)