import tensorflow as tf
import numpy as np

'''
RUN 'image_compression.py' INSTEAD TO SEE THE START-TO-FINISH AUTOENCODER

Running this script will only download a dataset, clean and preprocess it, and save it as a .npy file
'''

def grayscale(data, dtype='float32'):
    # luma coding, see https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems
    r, g, b = np.asarray(.299, dtype=dtype), np.asarray(.587, dtype=dtype), np.asarray(.114, dtype=dtype)
    luma = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    luma = np.expand_dims(luma, axis=3)
    return luma

def generate_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # can use mnist, cifar10 datasets from the keras repository
    # we only really need the x_train variable, the rest can be ignored

    print("Using TensorFlow ver.", tf.__version__)

    # TF requires 4D tensor in order: (samples, rows, cols, channels)
    samples = x_train.shape[0]
    image_rows = x_train.shape[1]
    image_cols = x_train.shape[2]

    if len(x_train.shape) == 4:  # grayscale only has 3 dims, no channel dimension
        channels = x_train.shape[3]
        if channels != 1:
            x_train = grayscale(x_train)

    print("Imported", samples, image_cols, "x", image_rows, "images.")

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(samples, 1, image_rows, image_cols)
        # x_test = x_test.reshape(samples, 1, image_rows, image_cols)
    else:
        x_train = x_train.reshape(samples, image_rows, image_cols, 1)
        # x_test = x_test.reshape(samples, image_rows, image_cols, 1)

    if tf.keras.backend.image_data_format() == 'channels_last':
        print("Reshaped input images to TensorFlow-compatible image tensor.")

    x_train = x_train.reshape(samples, image_rows*image_cols)
    print("Images converted from", image_cols, "x", image_rows, "array to", x_train.shape[1], "x 1 vector.")

    x_train = x_train.astype('float32') # for compatibility with tf
    x_train /= 255  # normalize

    return x_train, samples, (image_rows, image_cols)

if __name__ == '__main__':
    print("NOTE: To train or use the autoencoder, run 'image_compression.py' instead.")
    print("Running this script will only save the preprocessed data.\n")
    print("Please select the dataset in 'generate_data()'\n")

    input_data, samples, dims = generate_data()

    np.save('cleaned_input_data.npy', input_data)