import tensorflow as tf
import keras
from keras import layers, Model

@keras.saving.register_keras_serializable()
class CifarModel(Model):
    @staticmethod
    def residual_block(x, filters, filter_size=(3,3), stride=1, l2=1e-3, dropout=0.1):
        shortcut = x
        # First conv
        x = layers.Conv2D(filters, filter_size,
                          strides=stride,
                          padding="same",
                          kernel_regularizer=keras.regularizers.l2(l2),
                          activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SpatialDropout2D(dropout)(x)
        # Second conv
        x = layers.Conv2D(filters, filter_size,
                          padding="same",
                          kernel_regularizer=keras.regularizers.l2(l2),
                          activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SpatialDropout2D(dropout)(x)
        # Third conv
        x = layers.Conv2D(filters, filter_size,
                          padding="same",
                          kernel_regularizer=keras.regularizers.l2(l2),
                          activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SpatialDropout2D(dropout)(x)
        # Shortcut path
        if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
            shortcut = layers.Conv2D(filters, (1,1),
                                     strides=stride,
                                     padding="same",
                                     kernel_regularizer=keras.regularizers.l2(l2),
                                     activation=None)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        # Add & return
        x = layers.add([x, shortcut])
        return x

    def __init__(self, width=100):
        super().__init__()
        self.width = width

    def call(self, inputs, training=False):
        # Initial conv block
        x = layers.Conv2D(64, (3,3),
                          padding="same",
                          kernel_regularizer=keras.regularizers.l2(3e-5),
                          activation=None)(inputs)
        x = layers.BatchNormalization()(x, training=training)
        x = layers.ReLU()(x)
        x = layers.SpatialDropout2D(0.03)(x, training=training)

        # Residual blocks
        x = CifarModel.residual_block(x, filters=64, l2=3e-5, dropout=0.05, filter_size=(3,3), stride=1)
        x = layers.SpatialDropout2D(0.03)(x, training=training)

        x = CifarModel.residual_block(x, filters=128, l2=3e-5, dropout=0.05, filter_size=(3,3), stride=2)
        x = layers.SpatialDropout2D(0.05)(x, training=training)

        x = CifarModel.residual_block(x, filters=256, l2=3e-5, dropout=0.1, filter_size=(3,3), stride=2)
        x = layers.SpatialDropout2D(0.1)(x, training=training)

        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512,
                         activation=None,
                         kernel_regularizer=keras.regularizers.l2(5e-5))(x)
        x = layers.BatchNormalization()(x, training=training)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.5)(x, training=training)
        outputs = layers.Dense(10, activation="softmax")(x)
        return outputs