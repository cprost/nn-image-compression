import tensorflow as tf
import numpy as np
import cv2 # for displaying images
#from keras import backend

def generate_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # can use mnist, cifar10 datasets from the keras repository
    # we only really need the x_train variable, the rest can be ignored

    print("Using TensorFlow ver.", tf.__version__)

    # TF requires 4D tensor in order: (samples, rows, cols, channels)
    samples = x_train.shape[0]
    image_rows = x_train.shape[1]
    image_cols = x_train.shape[2]

    print("Imported", samples, image_cols, "x", image_rows, "images.")

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, image_rows, image_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, image_rows, image_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], image_rows, image_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], image_rows, image_cols, 1)

    if tf.keras.backend.image_data_format() == 'channels_last':
        print("Reshaped input images to TensorFlow-compatible image tensor.")

    x_train = x_train.reshape(60000, image_rows*image_cols)
    print("Images converted from", image_cols, "x", image_rows, "array to", x_train.shape[1],"x 1 vector.")

    x_train = x_train.astype('float32') # for compatibility with tf
    x_train /= 255 # normalize

    return x_train