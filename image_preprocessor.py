import tensorflow as tf
import cv2 # for displaying images
#from keras import backend

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# we only really need the x_train variable, the rest can be ignored

print("Imported TensorFlow ver.", tf.__version__)
print(x_train.shape)

# TF requires 4D tensor in order: (samples, rows, cols, channels)
image_rows = x_train.shape[1]
image_cols = x_train.shape[2]

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, image_rows, image_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, image_rows, image_cols)
    input_shape = (1, image_rows, image_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], image_rows, image_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], image_rows, image_cols, 1)
    input_shape = (image_rows, image_cols, 1)

print(x_train.shape)
if tf.kerasbackend.image_data_format() == 'channels_last':
    print("Reshaped input images to TensorFlow-compatible image tensor.")

x_train = x_train.reshape(60000,784)
print(x_train.shape)

img = cv2.imread('donut.jpg',0)
cv2.imshow('donut', img)
cv2.waitKey(0)