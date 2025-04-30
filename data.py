import numpy as np
import tensorflow as tf

class DataLoaderCifar:
    """Provide train, validation, and test datasets of the MNIST dataset."""

    def __init__(self, validation_dataset_size=5000, mini_batch_size=32, debug=True):

        cifar = tf.keras.datasets.cifar10
        train, test = cifar.load_data()
        
        self._train_dataset = train  # Use batching and shuffling
        self._valid_dataset = None
        self._test_dataset = test  # Use batching



    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset