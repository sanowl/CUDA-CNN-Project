import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def download_and_preprocess_mnist():
    # Download MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Reshape images to (28, 28, 1)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Save preprocessed data
    if not os.path.exists('data'):
        os.makedirs('data')
    
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)

    print("Data preprocessed and saved successfully.")

if __name__ == "__main__":
    download_and_preprocess_mnist()