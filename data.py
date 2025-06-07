import numpy as np
import tensorflow as tf

random_rotation = tf.keras.layers.RandomRotation(factor=1/24, fill_mode='reflect')

class DataLoaderCifar:
    """Provide train, validation, and test datasets of the MNIST dataset."""

    def __init__(self, validation_dataset_size=5000, mini_batch_size=32):
        
        cifar = tf.keras.datasets.cifar10
        train, test = cifar.load_data()

        train_images, train_labels = train
        train_images = normalize_images(train_images)
        full_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        valid_dataset = full_train.take(validation_dataset_size)
        train_dataset = full_train.skip(validation_dataset_size)
        
        augmented_dataset = train_dataset.map(
            augment_image, 
            num_parallel_calls=tf.data.AUTOTUNE
        )

        train_dataset = train_dataset.concatenate(augmented_dataset)

        test_images, test_labels = test
        test_images = normalize_images(test_images)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        
        self._train_dataset = (
            train_dataset.cache()
            .shuffle(train_dataset.cardinality())
            .cache()
            .batch(mini_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self._valid_dataset = (
            valid_dataset
            .cache()
            .batch(mini_batch_size)
        )
        self._test_dataset = test_dataset

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

def normalize_images(images):
    """normalizes images float32 [0,1]
    @type images: numpy.ndarray with dtype uint8
    """
    return tf.cast(images, tf.float32) / 255.0

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = random_rotation(image, training=True)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label